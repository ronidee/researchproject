import json
import itertools
import copy
import argparse
import jax

import jax.numpy as jnp
import numpy as np

from jax import jit
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from pathlib import Path

from differentiable import DiffableTree, tree_diff



DATASET_PATH = 'dataset/data_banknote_authentication.csv'

# Random seeds for reproducibility
seed(1) # cross validation splits
np.random.seed(1) # randomized sample

# TODO: Use np arrays instead of nested lists for the dataset
# Load the dataset file
def load_dataset():
    file = open(DATASET_PATH, "rt")
    dataset = list(reader(file))
    
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    return jnp.array(dataset)

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
            
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# TODO: check if diffrent split values have same gini index
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(jnp.unique(jnp.array(list(row[-1] for row in dataset))))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(jnp.unique(jnp.array(outcomes)).tolist(), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

def test_tree(tree, data):
    predictions = [predict(tree, sample) for sample in data]
    return accuracy_metric(data[:, -1], predictions)
    
def get_all_values(d):
    values = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(get_all_values(v))
        else:
            values.append(v)
    return values

# DANGEROUS: expects only 'b' to ever be traced by JAX
def same_type(a, b):
    # Check if a, b have same type or, if not and b is wrapped 
    # in a Tracer object, if a and b's wrapped variable have same type
    return type(a) == type(b) \
        or (isinstance(b, jax._src.interpreters.ad.JVPTracer) and type(a) == type(b.primal))


def randomize_sample(data, index):
    data = np.array(data) # convert to np incase its jnp, for in-place assignment
    # remove label column
    features = data[:, :-1]
    
    # for each feature, find lowest and highest value
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    
    random_features = np.random.uniform(low=feature_min, high=feature_max)

    # Randomly assign 1 or 0 to the label
    label = np.random.choice([0, 1])

    # Combine the features and label into a single data point
    random_data_point = np.append(random_features, label)
    data[index] = random_data_point
    return jnp.array(data)

def print_trees(t1, t2):
    print([json.dumps(t, indent=2) for t in (t1, t2)].join('\n'))

# simple function that prints the original and the reconstructed sample and their diff
def print_attack_stats():
    dummy_train = jnp.load("dummy_train.npy")
    print("Original sample:\t", client_train[attack_index])
    print("Reconst. sample:\t", dummy_train[attack_index])
    print("Original-Reconst.:\t", client_train[attack_index] - dummy_train[attack_index])


# diff_wrapper(dummy_train.tolist())

def init_client():
    # Create "original" client dataset and tree update
    client_train, client_test = train_test_split(dataset, train_size=args.n_train, test_size=args.n_test, random_state=args.rand_state)
    client_tree = DiffableTree(max_depth=args.max_depth, min_size=args.min_size)
    client_tree.fit(client_train.tolist())
    
    # Ask to save new client state    
    # user_input = input("Created new client dataset and tree upate. Save?[y/N]")
    # if user_input.lower() == 'y':
    #     pass

    return client_train, client_tree

def init_dummy(client_train, target_index):
    dummy_train = copy.deepcopy(client_train)
    dummy_train = randomize_sample(dummy_train, target_index)
    
    return dummy_train

def reconstruct_sample(args):   
    # Load client dataset and tree or create new one
    if args.client_state:
        pass
    else:
        client_train, client_tree = init_client()

    # Load dummy dataset or create new one
    if args.dummy_state:
        dummy_train = jnp.load(args.dummy_state.as_posix())
    else:
        dummy_train = init_dummy(copy.deepcopy(client_train), args.target_index)
    
    # Wrapper function to compute gradient on
    # computes diff between client tree and a tree that is freshly trained on dummy_train
    def diff_wrapper(_dummy_train):
        # train the dummy tree using '_dummy_train' data
        dummy_tree = DiffableTree(args.max_depth, args.min_size)
        dummy_tree.fit(_dummy_train)
        # print_trees(client_tree, dummy_tree)
        # exit()
        
        # compute the diff between the tree sent by the client and the dummy tree we just trained
        # this will be used to adapt the dummy data so that the new dummy_tree will resemble the client
        # tree more closely.
        d = tree_diff(client_tree, dummy_tree)
        # print(dummy_tree["left"]["value"].primal)
        print("tree diff =", d.primal)
        return d

    # TODO: check if all attributes (including label) change during the attack
    for i in range(4):
        # Compute the gradient of diff between both trees w.r.t. input (dummy_train)
        grad_diff = jnp.array(jax.grad(diff_wrapper)(dummy_train.tolist()))
        # print(f"sample diff (round {i}) = ", (dummy_train - client_train)[args.target_index])

        if not grad_diff.any():
            print("Attack completed!")
            break
        
        # update input by subtracting gradient
        dummy_train = dummy_train.at[args.target_index].set((dummy_train - 0.001*grad_diff)[args.target_index])
        jnp.save("out/dummy_states/dummy_train.npy", dummy_train)

def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    parser_attack = subparsers.add_parser("attack")
    parser_stats = subparsers.add_parser("stats")
    
    parser_attack.add_argument('--target-index', '-t', type=int, default=0, help="Index of sample to reconstruct.")
    parser_attack.add_argument('--client-state', '-c', type=Path, default=None, help="Existing client dataset and tree update.")
    parser_attack.add_argument('--dummy-state', '-d', type=Path, default=None, help="Existing dummy dataset (resume attack).")
    parser_attack.add_argument('--n-train', type=int, default=20, help="Size of new new client train set.")
    parser_attack.add_argument('--n-test', type=int, default=10, help="Size of new new client test set.")
    parser_attack.add_argument('--rand-state', type=int, default=42, help="Randomness for client's train/test split.")
    parser_attack.add_argument('--max-depth', '-u', type=int, default=5, help="Upper bound for tree depth during training.")
    parser_attack.add_argument('--min-size', '-l', type=int, default=2, help="Lower bound for no. samples after splits.")
    parser_attack.set_defaults(func=reconstruct_sample)
    
    parser_stats.add_argument('--dummy-state', '-d', type=Path, help="The dummy dataset to show the attack stats for.")
    parser_stats.set_defaults(func=print_attack_stats)

    return parser.parse_args()

if __name__ == "__main__":
    a = np.load("grad_diff_copy.npy")
    b = np.load("dummy_train_copy.npy")
    print(b)
    exit()
    # load and prepare data
    dataset = load_dataset()

    args = parse_arguments()
    args.func(args)


# scores = evaluate_algorithm(client_train.tolist()[:102], decision_tree, n_folds, max_depth, min_size)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores)))))