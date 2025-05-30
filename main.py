import argparse
from ast import arg
from matplotlib.pyplot import step
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import regression
import utils
import visualization


ABALONE_PATH = "dataset/abalone/abalone.data"

# Load specified dataset file
def load_dataset(dataset):
    if dataset == "abalone":
        encoder = LabelEncoder()
        data_abalone = pd.read_csv(ABALONE_PATH, header=None).values
        # the encoder seems to produce this mapping: (F,I,M) -> (0,1,2)
        data_abalone[:, 0] = encoder.fit_transform(data_abalone[:, 0])
        
        return data_abalone.astype(np.float64)
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
        client_tree = regression.RegressionTree(max_depth=args.max_depth, min_size=args.min_size)
        client_tree.fit(client_train)
        
        metrics = regression.test_tree(client_tree, client_test)
    
        # print(f"Built new client_tree (mse: {metrics['mse']:.3f}, mae: {metrics['mae']:.3f})")
        
        if utils.ask_yes_no("Created new client dataset and tree update. Save?"):
            state_dir = Path("out/iter")
            if not state_dir.exists():
                state_dir.mkdir()
            
            utils.save_client_state(state_dir=state_dir, client_train=client_train, client_tree=client_tree)
            print("Saved tree and dataset at " + state_dir.absolute().as_posix())    

    return client_train, client_test, client_tree

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
        # Base case: if this is a leaf node
        if not isinstance(node, dict):
            yield bounds
            return
    
        index = node.get('index', None)
        value = node.get('value', None)


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
    partitions = np.array(extract_partitions(client_tree.root, initial_bounds))
   
    for partition in partitions:
        # get training samples within partition
        leaf_samples_mask = intersect_with_partition(dummy_train[:, :-1], partition)
        # compare actual leaf node and mean of leaf_samples
        inferred_leaf_value = np.mean(dummy_train[leaf_samples_mask][:, -1]) if np.sum(leaf_samples_mask)>0 else 0.0
        actual_leaf_value = client_tree.predict(partition_center(partition))
        
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
    return round(leaf_info.size*leaf_info.actual - (leaf_info.size-1)*leaf_info.inferred)

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

def reconstruct_sample(args):
    utils.AUTOREPLY = args.autoreply
    # import random
    # random.seed(3)
    # np.random.seed(4)

    
    # create client training data and first update
    client_train, client_test, client_tree = init_client(args)
    
    
    # copy known part train part and init bounds with infinity
    dummy_train = np.delete(client_train, args.target_index, axis=0)
    derived_features = np.array([(-np.inf, np.inf)]*8)
    derived_features[0] = [0, 2+1e-8]
    derived_label = None
    # feature_bounds = utils.get_feature_bounds(client_train, extend_int_bounds=False)
    
    
    tree_depths = []
    for epoch in range(args.epochs):
        # 0. create new client tree update (yes, this wastes the first tree fit from init_client..)
        client_tree = regression.RegressionTree(args.max_depth, args.min_size)
        client_tree.fit(client_train)
        # print("client tree depth:", client_tree.final_depth)
        
        # include trees were leaf cannot be found, as this might be a result of depth and should be included in the count
        tree_depths.append(client_tree.final_depth)
        
        # 1. Identify the leaf the target sample ended up during training
        leaf_info = find_affected_leaf(client_tree, dummy_train)
        if not leaf_info:
            # print("couldnt find the leaf (target label must be equal to leaf value)")
            continue
        
        # 2. Derive label of target sample (only need to be done once)
        if derived_label == None:
            derived_label = derive_target_label(leaf_info)
            assert derived_label == client_train[args.target_index, -1]

        # 3. Determine the bounds for each feature of the target sample
        derived_features = derive_target_features(leaf_info=leaf_info, previous_bounds=derived_features)
        
    # utils.visualize_tree(tree1=client_tree, view=False, fp_out=Path("debuganalysis"))

    potential_samples_mask = intersect_with_partition(client_test[:, :-1], derived_features)
    potential_samples = client_test[potential_samples_mask]
    # potential_samples = potential_samples[potential_samples[:, -1] == derived_label]
    reconstructed_sample = partition_center(derived_features).round(3).tolist() + [derived_label]
    reconstructed_sample[0] = round(reconstructed_sample[0])
    
    np.set_printoptions(linewidth=np.inf, precision=3)

    within_bounds = (client_test[:, :-1] >= derived_features[:, 0]) & (client_test[:, :-1] < derived_features[:, 1])
    percent_within_bounds = within_bounds.sum(axis=0) / client_test.shape[0] * 100
    # print(f"""
    #       Potential samples in dataset: {potential_samples.shape[0]}
    #       Reconstructed: {reconstructed_sample}
    #       Value Bounds: {format_bounds(derived_features.round(3))}
    #       Actual sample: {client_train[args.target_index].tolist()}
    #       """)
    
    for sample in client_test[potential_samples_mask]:
        for i in range(8):
            assert sample[i] >= derived_features[i][0]
            assert sample[i] < derived_features[i][1]

    return {
        "candidates": potential_samples.shape[0],
        "avg_depth": sum(tree_depths) / len(tree_depths)
    } | {
        f"feature_{i}_percent": perc for i, perc in enumerate(percent_within_bounds)
    }


from tqdm import tqdm

def evaluate_attack(args):
    # 1. epochs vs. uniqueness
    args.min_size = 2
    args.max_depth = 15
    args.n_train = 500
    args.n_test = 3617
    args.rand_state = None
    args.autoreply = 'n'
    args.client_state = None
    args.dummy_state = None
    
    
    all_results = []
    for number_of_epochs in tqdm(range(1, 200, 8)):
        for _ in range(args.repeats):
            args.epochs = number_of_epochs
            
            result = reconstruct_sample(args)
            result["epochs"] = number_of_epochs
            all_results.append(result)
    
    df = pd.DataFrame(all_results)

    metrics = [col for col in df.columns if col != 'epochs']
    agg_results = df.groupby('epochs')[metrics].agg(['mean', 'std']).reset_index()
    agg_results.columns = ['_'.join(col).strip('_') for col in agg_results.columns.values]

    return agg_results, args.n_test
    

def create_figures(args):
    if "partition" in args.which:
        visualization.draw_tree_partition(args.show)
    if "splits" in args.which:
        visualization.plot_splits(args.show)
    if "evaluation" in args.which:
        visualization.plot_evaluation(*evaluate_attack(args), args.show)


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    parser.add_argument('--target-index', '-t', type=int, default=0, help="Index of sample to reconstruct")
    parser.add_argument('--dataset', '-D', choices=("abalone") , default="abalone", help="Name of dataset")
    
    parser_attack = subparsers.add_parser("attack")
    parser_stats = subparsers.add_parser("stats")
    parser_figures = subparsers.add_parser("figures")

    group_yes_no = parser_attack.add_argument_group("Default [y/n]-prompt reply", "Automatically answer prompts with either 'y' or 'n'")
    ex_group_yes_no = group_yes_no.add_mutually_exclusive_group()
    ex_group_yes_no.add_argument('--no', dest='autoreply', action='store_const', const='n', help="Always decline user prompts")
    ex_group_yes_no.add_argument('--yes', dest='autoreply', action='store_const', const='y', help="Always accept user prompts")
    parser_attack.add_argument('--learning-rate', '--lr', type=float, default=0.1, help="Initial learning rate for Adam")
    parser_attack.add_argument('--client-state', '-c', type=Path, default=None, help="Existing client dataset and tree update")
    parser_attack.add_argument('--dummy-state', '-d', type=Path, default=None, help="Existing dummy dataset (resume attack)")
    parser_attack.add_argument('--n-train', type=int, default=1000, help="Size of new new client train set")
    parser_attack.add_argument('--n-test', type=int, default=200, help="Size of new new client test set")
    parser_attack.add_argument('--rand-state', type=int, default=43, help="Randomness for client's train/test split")
    parser_attack.add_argument('--max-depth', '-u', type=int, default=15, help="Upper bound for tree depth during training")
    parser_attack.add_argument('--min-size', '-l', type=int, default=2, help="Lower bound for no. samples after splits")
    parser_attack.add_argument('--epochs', '-e', type=int, default=100, help="N.o. epochs to simulate the attack for (i.e. n.o. client updates)")
    parser_attack.set_defaults(func=reconstruct_sample) # callback function
    
    parser_stats.add_argument('--dummy-state', '-d', type=Path, help="The dummy dataset to show the attack stats for")
    parser_stats.add_argument('--client-state', '-c', type=Path, default=None, help="Existing client dataset and tree update")   
    parser_stats.set_defaults(func=print_attack_stats) # callback function

    parser_figures.add_argument('--which', '-w', choices=['evaluation', 'splits', 'partition'], required=True)
    parser_figures.add_argument('--repeats', '-r', type=int, default=5)
    parser_figures.add_argument('--show', '-s', action='store_true')
    parser_figures.set_defaults(func=create_figures)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    dataset = load_dataset(args.dataset)
    args.func(args)
