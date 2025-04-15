import jax
import json
import numpy
import jax.numpy as jnp
from jax.lax import cond
from jax.interpreters.ad import JVPTracer

from utils import smolhash


class JaxTracerEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, JVPTracer):
            return o.primal
        elif (isinstance(o, (jax.numpy.ndarray, numpy.ndarray)) and o.size == 1) or isinstance(o, numpy.float32):
            return o.item()
        elif isinstance(o, numpy.int32):
            return o.item()
        else:
            return o

# Check if a subset (selection of dataset in form of a boolean mask) is empty
def is_empty(mask):
    res = jnp.sum(mask) == 0
    return res


def diffable_subset_mean(arr, mask):
    mean = jnp.sum(jnp.where(mask, arr, 0.0)) / (jnp.sum(mask) + 1e-8)
    return mean


def mse(actual, predicted, xp=numpy):
    return xp.mean((actual - predicted) ** 2)


def mae(actual, predicted):
    return numpy.mean(numpy.abs(actual - predicted))


def test_tree(tree, data):
    predictions = numpy.array([tree.predict(sample) for sample in data[:, :-1]])
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
        if self.root and not retrain:
            raise AssertionError("Tree has already been trained. \
                If you're sure you want to fit it again, pass 'retrain=True'.")
        
        if not (self.max_depth and self.min_size):
            raise AssertionError("Attributes 'max_depth' and 'min_size' must be set before calling '.fit()'.")
        

        full_mask = self.xp.ones(train.shape[0], dtype=bool)
        self.root = DiffableTree.split(self, self.get_split(train, full_mask), 1, train)

        # after tree is formed, create fingerprint
        print("DANGER DANGER, USING GENERIC FINGERPRINT")
        self.fingerprint = "GENERIC" # self.generate_fingerprint(train)

    def get_split(self, dataset, mask):
        N, D = dataset[:, :-1].shape
        
        # Note: we want to try out *every* value for splitting at the current node (n.o. values=N*D)
        flat_indices = self.xp.repeat(self.xp.arange(D), N)  # shape: (D*N,)
        flat_values = dataset[:, :-1].T.flatten()  # shape: (D*N,)
        flat_mask = self.xp.repeat(mask[:, None].T, D, axis=0).flatten()

        def split_and_mse(index, value, mask_val):
            actual_result = lambda: (
                self.mse_calculator(*self.test_split(index, value, dataset, mask), dataset[:, -1]),
                index,
                value
            )
            placeholder_result = lambda: (self.xp.inf, -1, -1.0) # if 'value' is from sample not in current mask
            return cond(
                pred=mask_val, 
                true_fun=actual_result, 
                false_fun=placeholder_result
            )

        # TODO: better performance when doing self.xp.where outside split_and_mse()?
        mses, indices, values = jax.vmap(split_and_mse)(flat_indices, flat_values, flat_mask)

        # Find the best split
        best_idx = self.xp.argmin(mses)
        best_index = indices[best_idx]
        best_value = values[best_idx]
        best_groups = self.test_split(best_index, best_value, dataset, mask)

        return {
            'index': best_index,
            'value': best_value,
            'groups': best_groups,
        }
    
    def test_split(self, index, value, dataset, mask):
        # determine splitting masks and apply mask of current dataset subset (logical and)
        left_mask = dataset[:, index] < value
        return self.xp.logical_and(left_mask, mask), self.xp.logical_and(~left_mask, mask)

    @staticmethod
    def split(self, node, depth, dataset):
        left_mask, right_mask = node['groups']
        
        if depth >= self.max_depth:
            left_terminal = self.to_terminal(dataset, left_mask)
            right_terminal = self.to_terminal(dataset, right_mask)
            return {'left': left_terminal, 'right': right_terminal, 'index': node['index'], 'value': node['value']}
        
        if self.xp.sum(left_mask) <= self.min_size:
            left_node = self.to_terminal(dataset, left_mask)
        else:
            left_node = DiffableTree.split(self, self.get_split(dataset, left_mask), depth + 1, dataset)
        
        if self.xp.sum(right_mask) <= self.min_size:
            right_node = self.to_terminal(dataset, right_mask)
        else:
            right_node = DiffableTree.split(self, self.get_split(dataset, right_mask), depth + 1, dataset)
        
        return {'left': left_node, 'right': right_node, 'index': node['index'], 'value': node['value']}


    # Create a terminal node value
    def to_terminal(self, dataset, mask):
        return diffable_subset_mean(dataset[:, -1], mask)
    
    def mse_calculator(self, left_mask, right_mask, outcomes):
        def group_loss(mask):
            # Compute mean outcome of all samples in current group
            mean_outcome = diffable_subset_mean(outcomes, mask)
            
            squared_errors = self.xp.where(mask, (outcomes - mean_outcome)**2, 0.0)
            weighted_mse = self.xp.sum(squared_errors) # ("weighted" means multiply with group size. Hence mse -> se)
            return weighted_mse

        res = (group_loss(left_mask) + group_loss(right_mask)) / (self.xp.sum(left_mask) + self.xp.sum(right_mask))
        return res
    
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
    partitions, mask = get_intersected_partitions(client_tree, dummy_tree, initial_bounds)
    # partitions = jnp.unique(partitions, axis=0)
    partition_centers = partition_center(partitions)
    
    fn_squared_errors = jax.vmap(lambda x: (diffable_predict(client_tree.root, x) - diffable_predict(dummy_tree.root, x))**2)
    squared_errors_2 = fn_squared_errors(partition_centers)
    mse = jnp.sum(jnp.where(mask, squared_errors_2, 0.0)) / jnp.sum(mask)
    return mse


# Intersects each partition of the client tree with all partitions of the dummy tree
# a partition (shape=(8,2)) is an 8-D hyperrectangel. One tuple (lower, upper) for each dimension
def get_intersected_partitions(client_tree, dummy_tree, initial_bounds):
    client_partitions = jnp.array(extract_partitions(client_tree, initial_bounds))
    dummy_partitions = jnp.array(extract_partitions(dummy_tree, initial_bounds))

    client_partitions = jnp.unique(client_partitions, axis=0)
    dummy_partitions = jnp.unique(dummy_partitions, axis=0)

    # Compute the cartesian product indices
    num_cp = client_partitions.shape[0]
    num_dp = dummy_partitions.shape[0]
    
    # Create all index pairs (i,j)
    cp_idx, dp_idx = jnp.meshgrid(jnp.arange(num_cp), jnp.arange(num_dp), indexing='ij')
    cp_idx = cp_idx.flatten()
    dp_idx = dp_idx.flatten()

    # Extract all pairs
    cp_flat = client_partitions[cp_idx]
    dp_flat = dummy_partitions[dp_idx]

    # Vectorize the intersection function over these pairs
    intersections, mask = jax.vmap(intersect_partitions)(cp_flat, dp_flat)

    return intersections, mask


def intersect_partitions(r1, r2):
    """Compute intersection of two axis-aligned hyperrectangles."""
    lower = jnp.maximum(r1[:, 0], r2[:, 0])
    upper = jnp.minimum(r1[:, 1], r2[:, 1])
    
    is_not_empty = jnp.all(lower < upper)
    intersection = jnp.stack([lower, upper], axis=-1)
    
    return intersection, is_not_empty


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
