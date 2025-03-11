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

from differentiable import DiffableTree, tree_diff, test_tree
import utils



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
    
    acc = test_tree(client_tree, client_test)
    print(f"Built new client_tree with accuracy: {acc}")
    return client_train, client_tree

def init_dummy(client_train, target_index):
    dummy_train = copy.deepcopy(client_train)
    dummy_train = randomize_sample(dummy_train, target_index)
    
    return dummy_train
    
def reconstruct_sample(args):   
    # Load client dataset and tree or create new one
    if args.client_state:
        client_train, client_tree = utils.load_client_state(args.client_state)
    else:
        client_train, client_tree = init_client()
        user_input = input("Created new client dataset and tree upate. Save? [y/N]: ")
        if user_input.lower() == 'y':
            state_dir = Path("out/" + client_tree.fingerprint)
            
            if not state_dir.exists():
                state_dir.mkdir()
            
            utils.save_client_state(state_dir=state_dir, client_train=client_train, client_tree=client_tree)
            print("Saved tree and dataset at " + state_dir.absolute().as_posix())

    # Load dummy dataset or create new one
    if args.dummy_state:
        dummy_train = jnp.load(args.dummy_state.as_posix())
    else:
        dummy_train = init_dummy(copy.deepcopy(client_train), args.target_index)
        user_input = input("Created new dummy train data. Save? [y/N]: ")
        if user_input.lower() == 'y':
            fp_dummy_train = Path(f"out/dummy_train-n{args.n_train}-{utils.smolhash(dummy_train)}.npy")
            utils.save_dummy_state(fp_dummy_train, dummy_train)
    
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
    for i in range(10):
        # Compute the gradient of diff between both trees w.r.t. input (dummy_train)
        grad_diff = jnp.array(jax.grad(diff_wrapper)(dummy_train.tolist()))
        # print(f"sample diff (round {i}) = ", (dummy_train - client_train)[args.target_index])

        if not grad_diff.any():
            print("Attack completed!")
            break
        
        # update input by subtracting gradient
        dummy_train = dummy_train.at[args.target_index].set((dummy_train - 0.1*grad_diff)[args.target_index])
        jnp.save(f"out/dummy_states/dummy_train{i}.npy", dummy_train)

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
    parser_attack.add_argument('--rand-state', type=int, default=43, help="Randomness for client's train/test split.")
    parser_attack.add_argument('--max-depth', '-u', type=int, default=5, help="Upper bound for tree depth during training.")
    parser_attack.add_argument('--min-size', '-l', type=int, default=2, help="Lower bound for no. samples after splits.")
    parser_attack.set_defaults(func=reconstruct_sample)
    
    parser_stats.add_argument('--dummy-state', '-d', type=Path, help="The dummy dataset to show the attack stats for.")
    parser_stats.set_defaults(func=print_attack_stats)

    return parser.parse_args()

if __name__ == "__main__":
    # load and prepare data
    dataset = load_dataset()

    args = parse_arguments()
    args.func(args)
