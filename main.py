import json
import copy
import argparse
import jax
import optax

import jax.numpy as jnp
import numpy as np

from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from pathlib import Path

from differentiable import DiffableTree, tree_diff, test_tree
import utils
from utils import JaxTracerEncoder



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


def print_trees(*trees):
    print('\n'.join([json.dumps(t, indent=2, cls=JaxTracerEncoder) for t in trees]))

# simple function that prints the original and the reconstructed sample and their diff
def print_attack_stats(args):
    print("WARNING! USING STATIC TREE PARAMS FOR DUMMY TREE! TODO: STORE IN CONFIG")
    # dummy_sample = utils.load_dummy_state(state_dir=args.dummy_state)
    client_train, client_tree = utils.load_client_state(state_dir=args.client_state)
    # dummy_tree = DiffableTree(max_depth=5, min_size=2)
    
    # dummy_train = client_train.tolist()
    # dummy_train[0] = dummy_sample.tolist()
    
    # dummy_tree.fit(dummy_train)
    # print("Original sample:\t", client_train[0])
    # print("Reconst. sample:\t", dummy_sample)
    # print("Pairwise diff.:\t", client_train[0] - dummy_sample)
    
    utils.visualize_tree(client_tree.root, args.client_state / "client_tree", view=True)

# diff_wrapper(dummy_sample.tolist())

def init_client():
    # Create "original" client dataset and tree update
    client_train, client_test = train_test_split(dataset, train_size=args.n_train, test_size=args.n_test, random_state=args.rand_state)
    client_tree = DiffableTree(max_depth=args.max_depth, min_size=args.min_size)
    client_tree.fit(client_train.tolist())
    acc = test_tree(client_tree, client_test)
    print(f"Built new client_tree with accuracy: {acc}")
    
    return client_train, client_tree

def init_dummy(client_train):
    client_train = np.array(client_train) # convert to np incase its jnp, for in-place assignment
    # remove label column
    features = client_train[:, :-1]
    
    # for each feature, find lowest and highest value
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    
    # Create random features, label and append to one row
    random_features = np.random.uniform(low=feature_min, high=feature_max)
    random_label = np.random.choice([0, 1])
    random_data_point = np.append(random_features, random_label)
    
    return jnp.array(random_data_point)

def reconstruct_sample(args):
    state_dir = None
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


    # copy known client data (copy all, delete target)
    known_client_train = client_train.tolist()
    known_client_train[args.target_index] = None
    
    # Load dummy sample or create new one
    if args.dummy_state:
        dummy_sample = jnp.load(args.dummy_state.as_posix())
    else:
        dummy_sample = init_dummy(copy.deepcopy(client_train))
        user_input = input("Created new dummy train data. Save? [y/N]: ")
        if user_input.lower() == 'y':
            # store initial dummy sample (which is random) at iteration-0
            attack_state_dir = Path("out/" + client_tree.fingerprint).joinpath(f"target-{args.target_index}")
            test = attack_state_dir / "0/"
            utils.save_dummy_state(state_dir=attack_state_dir / "0", dummy_sample=dummy_sample)
    
    # Wrapper function to compute gradient on
    # computes diff between client tree and a tree that is freshly trained on dummy_train
    
    def diff_wrapper(_dummy_sample):
        # train the dummy tree using known part of client data and '_dummy_sample'
        dummy_train = known_client_train
        dummy_train[args.target_index] = _dummy_sample
        dummy_tree = DiffableTree(max_depth=args.max_depth, min_size=args.min_size)
        dummy_tree.fit(known_client_train)
        
        # compute the diff between the tree sent by the client and the dummy tree we just trained
        # this will be used to adapt the dummy data so that the new dummy_tree will resemble the client
        # tree more closely.
        d = tree_diff(client_tree, dummy_tree)
        print("tree diff =", d.primal)
        return d
    
    # Initialize Adam optimizer
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(dummy_sample)
    
    # TODO: check if all attributes (including label) change during the attack
    attack_state_dir = Path("out/") / client_tree.fingerprint / f"target-{args.target_index}"
    for i in range(500):
        # Compute the gradient of diff between both trees w.r.t. input (dummy_sample)
        # TODO: only calculate gradient for target sample row
        # TODO: split sample and label gradient calculation?
        grad_diff = jnp.array(jax.grad(diff_wrapper)(dummy_sample.tolist()))
        # print(f"sample diff (round {i}) = ", (dummy_sample - client_train)[args.target_index])
        # print("grad diff:", grad_diff)
        if not grad_diff.any():
            print("Attack completed!")
            break
        
        # update dummy_sample using adam optimizer
        updates, opt_state = optimizer.update(grad_diff, opt_state)
        updates = optax.apply_updates(dummy_sample, updates)
        dummy_sample = updates
        # dummy_sample = dummy_sample - args.learning_rate*grad_diff
        
        utils.save_dummy_state(state_dir=attack_state_dir / str(i+1), dummy_sample=dummy_sample)

def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    parser_attack = subparsers.add_parser("attack")
    parser_stats = subparsers.add_parser("stats")
    group_yes_no = parser_attack.add_argument_group("Default [y/n]-prompt reply", "Automatically answer prompts with either 'y' or 'n'")
    ex_group_yes_no = group_yes_no.add_mutually_exclusive_group()
    ex_group_yes_no.add_argument('--no', action='store_true', help="Always decline")
    ex_group_yes_no.add_argument('--yes', action='store_true', help="Always accept")
    parser_attack.add_argument('--learning-rate', '--lr', type=float, default=0.1, help="Initial learning rate for Adam")
    parser_attack.add_argument('--target-index', '-t', type=int, default=0, help="Index of sample to reconstruct")
    parser_attack.add_argument('--client-state', '-c', type=Path, default=None, help="Existing client dataset and tree update")
    parser_attack.add_argument('--dummy-state', '-d', type=Path, default=None, help="Existing dummy dataset (resume attack)")
    parser_attack.add_argument('--n-train', type=int, default=20, help="Size of new new client train set")
    parser_attack.add_argument('--n-test', type=int, default=10, help="Size of new new client test set")
    parser_attack.add_argument('--rand-state', type=int, default=43, help="Randomness for client's train/test split")
    parser_attack.add_argument('--max-depth', '-u', type=int, default=5, help="Upper bound for tree depth during training")
    parser_attack.add_argument('--min-size', '-l', type=int, default=2, help="Lower bound for no. samples after splits")
    parser_attack.set_defaults(func=reconstruct_sample)
    
    parser_stats.add_argument('--dummy-state', '-d', type=Path, help="The dummy dataset to show the attack stats for")
    parser_stats.add_argument('--client-state', '-c', type=Path, default=None, help="Existing client dataset and tree update")

    parser_stats.set_defaults(func=print_attack_stats)

    return parser.parse_args()

if __name__ == "__main__":
    # load and prepare data
    dataset = load_dataset()

    args = parse_arguments()
    args.func(args)
