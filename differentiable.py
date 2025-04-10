import jax
import json
import numpy
import itertools
import pandas as pd
from jax.interpreters.ad import JVPTracer

from utils import smolhash


class JaxTracerEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, JVPTracer):
            return o.primal
        elif isinstance(o, jnp.ndarray) and o.size == 1 or isinstance(o, numpy.float32):
            return o.item()
        else:
            return o


# This isn't actually large. But large in the context of tree diff errors (at least I hope so...)
VERY_LARGE_NUMBER = 10**0

# Calculate accuracy percentage
def mse(actual, predicted, xp=numpy):
    return xp.mean((actual - predicted) ** 2)

def mae(actual, predicted):
    return numpy.mean(numpy.abs(actual - predicted))

def test_tree(tree, data):
    predictions = [tree.predict(sample) for sample in data[:, :-1]]
    
    # # Create a readable table
    # results = pd.DataFrame({
    #     'Actual': data[:, -1],
    #     'Predicted': predictions,
    #     'Error': data[:, -1] - predictions
    # })
    # print(results.head(10).to_string(index=False)) 
    return {
        'mse': mse(data[:, -1], predictions),
        'mae': mae(data[:, -1], predictions)
    }


class DiffableTree:
    def __init__(self, max_depth=None, min_size=None, root=None, trace=True, fingerprint=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = root
        self.trace = trace
        
        if root:
            assert fingerprint # loading trained tree requires original fingerprint
            self.fingerprint = fingerprint
        
    
    @property
    def xp(self):
        return jax.numpy if self.trace else numpy
    
    
    # Wrapper for non-instance __predict() function
    def predict(self, row):
        return self.__predict(self.root, row)

    
    # Make a prediction with a decision tree
    def __predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.__predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.__predict(node['right'], row)
            else:
                return node['right']


    # Fit tree to 'train' data, which contains labels at last column
    def fit(self, train, retrain=False):
        assert type(train) != list
        
        if self.root and not retrain:
            raise AssertionError("Tree has already been trained. \
                If you're sure you want to fit it again, pass 'retrain=True'.")
        
        if not (self.max_depth and self.min_size):
            raise AssertionError("Attributes 'max_depth' and 'min_size' must be set before calling '.fit()'.")

        self.root = self.get_split(train)
        self.split(self.root, 1)
        
        # after tree is formed, create fingerprint
        self.fingerprint = self.generate_fingerprint(train)


    def get_split(self, dataset):
        best_index, best_value, best_mse, best_groups = None, None, float('inf'), None
        for index in range(dataset.shape[1] - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                mse = self.mse_calculator(groups, dataset.shape[0])
                if mse < best_mse:
                    best_index, best_value, best_mse, best_groups = index, row[index], mse, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    
    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        mask = dataset[:, index] < value
        return dataset[mask], dataset[~mask]

    
    def split(self, node, depth):
        left, right = node.pop('groups')

        # check for a no split
        if left.size == 0 or right.size == 0:
            node['left'] = node['right'] = self.to_terminal(self.xp.vstack((left, right)))
            return
        
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return

        # process left child
        node['left'] = self.to_terminal(left) if len(left) <= self.min_size else self.get_split(left)
        if isinstance(node['left'], dict):
            # not a terminal node, keep processing
            self.split(node['left'], depth + 1)

        # process right child
        node['right'] = self.to_terminal(right) if len(right) <= self.min_size else self.get_split(right)
        if isinstance(node['right'], dict):
            # not a terminal node, keep processing
            self.split(node['right'], depth + 1)


    # Create a terminal node value
    def to_terminal(self, group):
        if len(group) == 0:
            raise ValueError("argument 'group' can't be empty!")
        return self.xp.mean(group[:, -1])


    def mse_calculator(self, groups, n_samples):
        mse_split = 0
        # calculate mse for both split sides, weighted by respective group size
        for group in groups:
            if group.size == 0: continue
            weighted_mse = mse(group[:, -1], self.to_terminal(group), xp=self.xp) * group.shape[0]
            mse_split += (weighted_mse) / n_samples

        return mse_split
    
    
    # Check whether this tree instance is equal to tree instance 'tree2'
    def equals(self, tree2):
        return tree_diff(self, tree2) == 0

    
    # create fingerprint from train params and tree hash
    # trees with different fingerprints may still satisfy the .equals() function
    def generate_fingerprint(self, train):
        # create hash by hashing the concatenated dict and ds hashes
        root_hash = smolhash(self)
        train_hash = smolhash(self.xp.array(jax.lax.stop_gradient(train))) # stop gradient during grad calculation
        tree_hash = smolhash(root_hash + train_hash)
        
        return f"u{self.max_depth}-l{self.min_size}-{tree_hash}"


    def to_json(self):
        return json.dumps(
            obj={
                "root": self.root,
                "max_depth": self.max_depth,
                "min_size": self.min_size,
                "trace": self.trace,
                "fingerprint": self.fingerprint
            },
            indent=2,
            cls=JaxTracerEncoder
        )
    
    
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)


import jax.numpy as jnp

def soft_step(x, threshold, steepness=100.0):
    # Approximates the indicator function x < threshold using a sigmoid.
    return jax.nn.sigmoid(steepness * (threshold - x))

def diffable_predict(node, sample, steepness=100.0):
    """
    Recursively computes a soft prediction.
    Instead of a hard branch, we use the sigmoid to softly weight left/right predictions.
    """
    # Base case: if the node is a terminal, return its value.
    if not isinstance(node, dict):
        return node
    # Get the split parameters.
    index = node['index']
    threshold = node['value']
    
    # Compute soft decision weight for left branch (close to 1 if sample[index] < threshold)
    left_weight = soft_step(sample[index], threshold, steepness)
    # The right branch gets the complementary weight.
    right_weight = 1.0 - left_weight
    
    # Recursively compute the predictions for left and right branches.
    left_pred = diffable_predict(node['left'], sample, steepness)
    right_pred = diffable_predict(node['right'], sample, steepness)
    
    # Return a weighted average.
    return left_weight * left_pred + right_weight * right_pred


# this function compares the outputs of each tree for the entire input space
# this is done by analyzing the partitions both trees create and superimposing them
# each (new) cell is then evaluated by picking the sample in its center
def compute_tree_difference(client_tree, dummy_tree, initial_bounds):
    # root1 = {
    #     "index": 0,
    #     "value": 8,
    #     "left": 1,
    #     "right": {
    #         "index": 0,
    #         "value": 14,
    #         "left": {
    #             "index": 1,
    #             "value": 50,
    #             "left": 0,
    #             "right": {
    #                 "index": 1,
    #                 "value": 100,
    #                 "left": 1,
    #                 "right": 0
    #             }   
    #         },
    #         "right": 0
    #     }
    # }
    # root2 = {
    #     "index": 0,
    #     "value": 5,
    #     "left": 1,
    #     "right": 0
    # }
    # tree1 = DiffableTree(root=root1)
    # tree2 = DiffableTree(root=root1)
    # samples = [
    #     [7.9, 0],   # 1
    #     [7.9, 200], # 1
    #     [5, 54],    # 1
    #     [10, 49],   # 0
    #     [10, 51],   # 1
    #     [10, 101],  # 0
    #     [10, 2002], # 0
    #     [15, 10],   # 0
    #     [15, 55],   # 0
    #     [15, 150],  # 0
    #     [15, 500],  # 0
    #     ]
    # samples = jnp.array(samples)
    # predictions = [tree1.predict(sample) for sample in samples]
    # assert [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0] == predictions
    partitions = get_refined_partitions(client_tree, dummy_tree, initial_bounds)
    partitions = jnp.unique(partitions, axis=0)
    partition_centers = partition_center(partitions)
    # centers_printable = "\n".join(" / ".join(str(feat_val) for feat_val in center) for center in partition_centers)
    # print(f"partition_centers ({len(partition_centers)}):", "\n" + centers_printable)
    # predictions = jnp.array([[client_tree.predict(x), dummy_tree.predict(x)] for x in partition_centers])
    predictions = jnp.array([[diffable_predict(client_tree.root, x), diffable_predict(dummy_tree.root, x)] for x in partition_centers])
    return mse(*predictions.T, xp=jnp)
#4.3757496
def get_refined_partitions(client_tree, dummy_tree, initial_bounds):
    client_partitions = jnp.array(extract_partitions(client_tree, initial_bounds))
    dummy_partitions = jnp.array(extract_partitions(dummy_tree, initial_bounds))
    
    # exit()
    client_partitions = jnp.unique(client_partitions, axis=0)
    dummy_partitions = jnp.unique(dummy_partitions, axis=0)

    refined_regions = []
    for cr in client_partitions:
        for dr in dummy_partitions:
            region = intersect_regions(cr, dr)
            if region is not None:
                refined_regions.append(region)
    return jnp.array(refined_regions)


def intersect_regions(r1, r2):
    """Compute intersection of two axis-aligned hyperrectangles."""
    lower = jnp.maximum(r1[:, 0], r2[:, 0])
    upper = jnp.minimum(r1[:, 1], r2[:, 1])
    if jnp.any(lower >= upper):
        return None  # Empty intersection
    return jnp.stack([lower, upper], axis=-1)

def extract_partitions(tree, initial_bounds):
    """
    Recursively extract all decision cells from a DiffableTree.
    Each region is represented as a list of (lower, upper) bounds for each feature.
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

    return list(recurse(tree.root, initial_bounds))


def partition_center(arr):
    assert 2 <= arr.ndim <= 3 # only allow single cell or array of cells
    return jnp.mean(arr, axis=arr.ndim-1)
    

def count_features(node):
    if node is None:
        return 0
    if isinstance(node, dict):
        return max(
            node['index'] + 1,
            count_features(node.get('left')),
            count_features(node.get('right')),
        )
    return 0





# --- Extract Splitting Boundaries from a Tree ---
def extract_boundaries(node, feature_index, boundaries=None):
    if boundaries is None:
        boundaries = []
    if isinstance(node, dict):
        if node['index'] == feature_index:
            boundaries.append(node['value'])
        # Recursively extract from left and right.
        extract_boundaries(node['left'], feature_index, boundaries)
        extract_boundaries(node['right'], feature_index, boundaries)
    return boundaries



# computes diff between two trees by summed squared distance of indexes/thresholds.
# Different structure (leaf vs non-leaf) inflicts instant 'VERY_LARGE_NUMBER' damage
# @param level: level in the tree hierachy (root node=0), so we know when to sum up. Could replace by always summing up. TODO!
# TODO: remove dependence on same random seed during training
def tree_diff(tree1, tree2, level=0):#client_test, level=0):
    # preds_1 = [tree1.predict(sample) for sample in client_test]
    # preds_2 = [tree2.predict(sample) for sample in client_test]

    # print(preds_1, "\n", preds_2)
    # print(np.sum(np.array(preds_1) - np.array(preds_2)))
    # exit()

    # extract tree structure (dict) from instance
    if isinstance(tree1, DiffableTree): tree1 = tree1.root
    if isinstance(tree2, DiffableTree): tree2 = tree2.root
    
    # list of errors to be summed up with jnp.sum(). Could probably use +=, but I'll check that later
    errors = []
    # Here, v1/v2 are values of fields "index", "value", "left", "right"
    for v1, v2 in zip(tree1.values(), tree2.values()):
        # this indicates, that one tree has a leaf node, while the other has not
        if not same_type(v1, v2):
            errors.append(VERY_LARGE_NUMBER * (1/(level+1)))
            
            if type(v1) == dict:
                # replace v2 by dummy tree to resume traversing v1
                v2 = dict(enumerate([None] * 4)) # dummy tree
            else:
                # replace v2 with v1, so the error will be 0, since we already added the VERY_LARGE_NUMBER penalty
                v2 = v1

        if isinstance(v1, dict): # calculate difference for the subtree
            errors.extend(tree_diff(v1, v2, level=level+1))
        elif isinstance(v1, int): # calculate difference for index selection
            errors.append(int(v1 != v2) * VERY_LARGE_NUMBER * (1/(level+1))) # Note: For high feature counts, this could cause overshoots
        elif isinstance(v1, float):
            errors.append(jax.numpy.pow(v1-v2, 2))
    
    # return sum of aggregated errors
    return jax.numpy.sum(jax.numpy.array(errors)) if level == 0 else errors

    
# DANGEROUS: expects only 'b' to ever be traced by JAX
def same_type(a, b):
    # Check if a, b have same type or, if not and b is wrapped 
    # in a Tracer object, if a and b's wrapped variable have same type
    if (isinstance(b, jax._src.interpreters.ad.JVPTracer)):
        if isinstance(a, dict): return False
        if isinstance(a, (numpy.floating, numpy.ndarray)): return type(a.item()) == type(b.primal.item())
        
    return type(a) == type(b)