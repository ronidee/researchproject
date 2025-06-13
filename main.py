from time import perf_counter
import optax
import argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import iteratively
import utils
from differentiable import compute_tree_difference

np.random.seed(2)


ABALONE_PATH = "dataset/abalone/abalone.data"

# Load specified dataset file
def load_dataset(dataset):
    # TODO: normalize features
    if dataset == "abalone":
        encoder = LabelEncoder()
        data_abalone = pd.read_csv(ABALONE_PATH, header=None).values
        # the encoder seems to produce this mapping: (F,I,M) -> (0,1,2)
        data_abalone[:, 0] = encoder.fit_transform(data_abalone[:, 0])
        
        return data_abalone.astype(np.float32)
    else:
        raise ValueError("Argument 'dataset' must be 'abalone'. Argparse should've enforced this.")


# simple function that prints the original and the reconstructed sample and their diff
def print_attack_stats(args):
    dummy_sample = dummy_tree = client_train = client_tree = None
    if args.dummy_state:
        dummy_sample, dummy_tree = utils.load_dummy_state(state_dir=args.dummy_state)
    if args.client_state:
        client_train, client_tree = utils.load_client_state(state_dir=args.client_state)
    
    if isinstance(client_train, np.ndarray):
        print("Original sample:\t", client_train[args.target_index])
    if isinstance(dummy_sample, np.ndarray):
        print("Reconst. sample:\t", dummy_sample)
    if isinstance(client_train, np.ndarray) and isinstance(dummy_sample, np.ndarray):
        print("Pairwise diff.:\t", client_train[args.target_index] - dummy_sample)
    
    fp_out = (
        args.dummy_state / "client_vs_dummy" if client_tree and dummy_tree else
        args.client_state / "client_tree"    if client_tree else
        args.dummy_state / "dummy_tree"      if dummy_tree else None
    )
    
    if fp_out:
        utils.visualize_tree(tree1=client_tree, tree2=dummy_tree, fp_out=fp_out, view=True)

def init_client(args):
    # Load client dataset and tree or create new one
    if args.client_state:
        client_train, client_tree = utils.load_client_state(args.client_state)
    else:
        # Create "original" client dataset and tree update
        client_train, client_test = train_test_split(dataset, train_size=args.n_train, test_size=args.n_test, random_state=args.rand_state)
        client_tree = iteratively.train_tree(
            client_train,
            max_depth=args.max_depth,
            min_size=args.min_size,
        )
        
        metrics = iteratively.test_tree(client_tree, client_test)
    
        print(f"Built new client_tree (mse: {metrics['mse']:.3f}, mae: {metrics['mae']:.3f})")
        
        if utils.ask_yes_no("Created new client dataset and tree update. Save?"):
            state_dir = Path("out/iter")
            if not state_dir.exists():
                state_dir.mkdir()
            
            utils.save_client_state(state_dir=state_dir, client_train=client_train, client_tree=client_tree)
            print("Saved tree and dataset at " + state_dir.absolute().as_posix())    

    return client_train, client_tree

@dataclass
class LeafInfo:
    actual: float
    inferred: float
    size: int
    partition: np.ndarray

def extract_partitions(tree, initial_bounds):
    """
    Recursively extract all decision cells from a RegressionTree.
    Each partitions is represented as a list of (lower, upper) bounds for each feature.
    """
    def recurse(node, bounds):
        index = node.get('index', None)
        value = node.get('value', None)

        # Base case: if this is a leaf node
        if iteratively.is_leaf(node):
            yield bounds
            return

        # If left child exists
        left_bounds = [b if i != index else (b[0], min(b[1], value)) for i, b in enumerate(bounds)]
        if not iteratively.is_leaf(node['left']):
            yield from recurse(node['left'], left_bounds)
        else:
            yield left_bounds

        # If right child exists
        right_bounds = [b if i != index else (max(b[0], value), b[1]) for i, b in enumerate(bounds)]
        if not iteratively.is_leaf(node['right']):
            yield from recurse(node['right'], right_bounds)
        else:
            yield right_bounds

    return list(recurse(tree, initial_bounds))

def extract_partitions_old(tree, initial_bounds):
    """
    Recursively extract all decision cells from a RegressionTree.
    Each partitions is represented as a list of (lower, upper) bounds for each feature.
    """
    def recurse(node, bounds):
        index = node.get('index', None)
        value = node.get('value', None)

        # Base case: if this is a leaf node
        if not isinstance(node, dict):
            yield bounds
            return

        # If left child exists
        left_bounds = [b if i != index else (b[0], min(b[1], value)) for i, b in enumerate(bounds)]
        if isinstance(node['left'], dict):
            yield from recurse(node['left'], left_bounds)
        else:
            yield left_bounds

        # If right child exists
        right_bounds = [b if i != index else (max(b[0], value), b[1]) for i, b in enumerate(bounds)]
        if isinstance(node['right'], dict):
            yield from recurse(node['right'], right_bounds)
        else:
            yield right_bounds

    return list(recurse(tree, initial_bounds))

def intersect_partitions(r1, r2):
    """Compute intersection of two axis-aligned hyperrectangles."""
    lower = np.maximum(r1[:, 0], r2[:, 0])
    upper = np.minimum(r1[:, 1], r2[:, 1])
    if np.any(lower >= upper):
        
        return None  # Empty intersection
    return np.stack([lower, upper], axis=-1)

# returns all elements in 'samples' that are inside given 'partition'
def intersect_with_partition(samples, partition):
    within_lower = samples >= partition[:, 0]
    within_upper = samples < partition[:, 1]
    within_bounds = np.logical_and(within_lower, within_upper)
    within_partition = np.all(within_bounds, axis=1)
    return within_partition

def partition_center(partition):
    assert 2 <= partition.ndim <= 3 # only allow single cell or array of cells

    # replace (-inf, +inf) rows with (-1, -1) for safe mean computation
    # value is irrelevant, as (-inf, +inf) means .predict() doesn't use the feature
    unsafe_mask = (partition[:, 0] == -np.inf) & (partition[:, 1] == np.inf)
    safe_partition = np.copy(partition)
    safe_partition[unsafe_mask] = [-1, -1] 
    
    return np.mean(safe_partition, axis=partition.ndim-1)

def find_affected_leaf(client_tree, dummy_train):
    # find all partitions
    initial_bounds = np.array([(-np.inf, np.inf)]*8)
    partitions = np.array(extract_partitions(client_tree, initial_bounds))
    for partition in partitions:
        # get training samples within partition
        leaf_samples_mask = intersect_with_partition(dummy_train[:, :-1], partition)
        # compare actual leaf node and mean of leaf_samples
        inferred_leaf_value = np.mean(dummy_train[leaf_samples_mask][:, -1]) if np.sum(leaf_samples_mask)>0 else 0.0
        actual_leaf_value = iteratively.predict(client_tree, partition_center(partition))
        
        # if it doesn't match the actual leaf node, we found the correct leaf
        if inferred_leaf_value != actual_leaf_value:
            # return leaf value and n.o. samples it was derived (mean) from
            return LeafInfo(
                actual=actual_leaf_value,
                inferred=inferred_leaf_value,
                size=np.sum(leaf_samples_mask)+1,
                partition=partition
            )

def derive_target_label(leaf_info):
    return leaf_info.size*leaf_info.actual - (leaf_info.size-1)*leaf_info.inferred

def derive_target_features(leaf_info, previous_bounds):
    # combine previously known bounds with new found ones
    new_feature_bounds = intersect_partitions(previous_bounds, leaf_info.partition)
    assert new_feature_bounds is not None, "I strongly believe that's impossible"
    return new_feature_bounds
    # return np.concat((partition_center(new_feature_bounds), [target_label]))

def format_bounds(bounds_array):
    formatted = []
    long_dash = '–'  # en dash (U+2013), used for ranges like 2.1–4.5
    less_equal = '≤'
    
    for lower, upper in bounds_array:
        if np.isneginf(lower) and np.isposinf(upper):
            formatted.append("?")  # or skip / add custom text
        elif np.isneginf(lower):
            formatted.append(f"<{upper}")
        elif np.isposinf(upper):
            formatted.append(f"{lower}{less_equal}")
        else:
            formatted.append(f"{lower}{long_dash}{upper}")

    return formatted

def init_dummy(client_train):
    # remove labels
    features = client_train[:, :-1]
    
    # find bounds of each feature (to improve init dummy sample. Helpful? Reasonable assumption?)
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    
    # generate random sample within retrieved feature bounds
    random_features = np.random.uniform(low=feature_min, high=feature_max)
    random_label = np.random.choice([4, 18])
    # print("WARNING: replacing random label with target label!")
    random_data_point = np.append(random_features, random_label)
    
    return np.array(random_data_point)

def reconstruct_sample(args):
    utils.AUTOREPLY = args.autoreply
    ts = perf_counter()
    
    # create client training data and first update
    client_train, client_tree = init_client(args)
    
    # copy known part train part and init bounds with infinity
    known_client_train = jnp.delete(client_train, args.target_index, axis=0)
    feature_bounds = utils.get_feature_bounds(client_train)
    
    dummy_sample = jnp.array([0.8719898,   0.42725933,  0.46768081,  0.14135563,  0.9600838,   0.40061867,  0.15064119,  0.34395811, 18.])#init_dummy(known_client_train)
    # utils.visualize_tree(tree1=tree, tree2=client_tree, view=True, labels=("recursive", "iterative"), fp_out=Path("debuganalysis"))
   
    def diff_wrapper(_dummy_sample):#, _attack_snapshot_dir=None):
        # train the dummy tree using known part of client data and '_dummy_sample'
        dummy_train = jnp.vstack((_dummy_sample, known_client_train))
        dummy_tree = iteratively.train_tree(
            dummy_train,
            max_depth=args.max_depth,
            min_size=args.min_size
        )
        # utils.save_dummy_state(state_dir=_attack_snapshot_dir, dummy_tree=dummy_tree, ignore_conflicts=True)
        
        # compute the diff between the tree sent by the client and the dummy tree we just trained
        # this will be used to adapt the dummy data so that the new dummy_tree will resemble the client
        # tree more closely.
        d = compute_tree_difference(client_tree, dummy_tree, client_train[:, -1], dummy_train[:, -1], feature_bounds)
        print("tree diff =", d.primal)
        return d
    
    # Initialize Adam optimizer
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(dummy_sample)

    jnp.set_printoptions(linewidth=np.inf)
    np.set_printoptions(linewidth=np.inf)
    dummy_iterations = []

    # attack_state_dir = Path("out/") / client_tree.fingerprint / f"target-{args.target_index}"
    for i in range(10):
        # tree_snapshot_dir = attack_state_dir / str(i) # based on previous dummy_sample
        # sample_snapshot_dir = attack_state_dir / str(i+1) # +1, since random is "/0/"
        
        # TODO: split sample and label gradient calculation?
        # Compute the gradient of diff between both trees w.r.t. input (dummy_sample)
        
        # print("Do stop_gradient after each iteration? I think its not needed, as outside of grad nothing is being traced.. right?")
        # dummy_sample = jax.lax.stop_gradient(dummy_sample)
        grad_diff = jax.grad(diff_wrapper)(dummy_sample)#, tree_snapshot_dir)
        if not grad_diff.any():
            print("Attack completed!")
            break

        # update dummy_sample using adam optimizer
        updates, opt_state = optimizer.update(grad_diff, opt_state)
        updates = optax.apply_updates(dummy_sample, updates)
        dummy_sample = updates
        dummy_iterations.append(dummy_sample.copy())
    
    for sample in dummy_iterations:
        print("target index:", client_train[args.target_index])
        print("dummy sample:", sample, "\n")

def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    parser.add_argument('--target-index', '-t', type=int, default=0, help="Index of sample to reconstruct")
    parser.add_argument('--dataset', '-D', choices=("abalone") , default="abalone", help="Name of dataset")
    
    parser_attack = subparsers.add_parser("attack")
    parser_stats = subparsers.add_parser("stats")
    group_yes_no = parser_attack.add_argument_group("Default [y/n]-prompt reply", "Automatically answer prompts with either 'y' or 'n'")
    ex_group_yes_no = group_yes_no.add_mutually_exclusive_group()
    ex_group_yes_no.add_argument('--no', dest='autoreply', action='store_const', const='n', help="Always decline user prompts")
    ex_group_yes_no.add_argument('--yes', dest='autoreply', action='store_const', const='y', help="Always accept user prompts")
    parser_attack.add_argument('--learning-rate', '--lr', type=float, default=0.1, help="Initial learning rate for Adam")
    parser_attack.add_argument('--client-state', '-c', type=Path, default=None, help="Existing client dataset and tree update")
    parser_attack.add_argument('--dummy-state', '-d', type=Path, default=None, help="Existing dummy dataset (resume attack)")
    parser_attack.add_argument('--n-train', type=int, default=100, help="Size of new new client train set")
    parser_attack.add_argument('--n-test', type=int, default=10, help="Size of new new client test set")
    parser_attack.add_argument('--rand-state', type=int, default=43, help="Randomness for client's train/test split")
    parser_attack.add_argument('--max-depth', '-u', type=int, default=5, help="Upper bound for tree depth during training")
    parser_attack.add_argument('--min-size', '-l', type=int, default=2, help="Lower bound for no. samples after splits")
    parser_attack.add_argument('--epochs', '-e', type=int, default=10, help="N.o. epochs to simulate the attack for (i.e. n.o. client updates)")
    parser_attack.set_defaults(func=reconstruct_sample) # callback function
    
    parser_stats.add_argument('--dummy-state', '-d', type=Path, help="The dummy dataset to show the attack stats for")
    parser_stats.add_argument('--client-state', '-c', type=Path, default=None, help="Existing client dataset and tree update")   
    parser_stats.set_defaults(func=print_attack_stats) # callback function

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    dataset = load_dataset(args.dataset)
    args.func(args)
