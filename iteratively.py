import numpy as np
from random import randrange
import typing
from dataclasses import dataclass


@dataclass
class SplitInfo:
    index: np.int8
    value: np.float32
    left_mask: np.ndarray
    right_mask: np.ndarray
    
    def __post_init__(self):        
        assert isinstance(self.index, np.int8), "TypeError for field 'index'"
        assert isinstance(self.value, np.float32), "TypeError for field 'value'"
        assert isinstance(self.left_mask, np.ndarray) and self.left_mask.dtype==bool , "TypeError for field 'left_mask'"
        assert isinstance(self.right_mask, np.ndarray) and self.right_mask.dtype==bool , "TypeError for field 'right_mask'"

        left_size = np.sum(self.left_mask)
        right_size = np.sum(self.right_mask)
        self.is_pseudo_split = left_size == 0 or right_size == 0

# random.seed(1)

__feature_bounds = None


def cond(pred, true_fun, false_fun):
  if pred:
    return true_fun()
  else:
    return false_fun()


def train_tree(train, max_depth=None, min_size=None, feature_bounds=None):
    global __feature_bounds
    __feature_bounds = feature_bounds
    
    def too_small(_node):
        return np.sum(_node['subset_mask']) <= min_size
    
    root = {'subset_mask': np.ones(train.shape[0], dtype=bool)}
    raw_nodes = [root]
    for depth in range(0, max_depth+1):
        new_raw_nodes = [] # raw nodes for next iteration

        # iterate over all nodes in current level/depth and split/leafify them
        for node in raw_nodes:
            if too_small(node) or depth == max_depth:
                to_leaf(node, train[:, -1])
            else:
                child_nodes = split(node, train, return_childs=True)
                new_raw_nodes.extend(child_nodes)

        # update raw_nodes for next iteration
        raw_nodes = new_raw_nodes

    return root



def split(node, train, return_childs=False):
    child_nodes = []
    subset_mask = node['subset_mask']
    split_info = find_best_split(subset_mask, train)
    
    if split_info.is_pseudo_split:
        to_leaf(node, train[:, -1])
    else:
        del node['subset_mask']
        node['index'] = split_info.index
        node['value'] = split_info.value
        node['left'] = {'subset_mask': split_info.left_mask}    
        node['right'] = {'subset_mask': split_info.right_mask}
        child_nodes = [node['left'], node['right']]
    
    if return_childs:
        return child_nodes


def to_leaf(node, outcomes):
    # delete subset_mask key and add prediction
    mean_outcome = diffable_subset_mean(outcomes, node.pop('subset_mask'))
    node['prediction'] = mean_outcome


def diffable_subset_mean(arr, subset_mask):
    if np.sum(subset_mask) == 0:
        return 0
    
    mean = np.sum(np.where(subset_mask, arr, 0.0)) / (np.sum(subset_mask))
    return mean


def find_best_split(mask, train):
    N, D = train[:, :-1].shape
    
    # Note: we want to try out *every* value for splitting at the current node (n.o. values=N*D)
    flat_indices = np.repeat(np.arange(D), N)  # shape: (D*N,)
    flat_values = train[:, :-1].T.flatten()  # shape: (D*N,)
    flat_mask = np.repeat(mask[:, None].T, D, axis=0).flatten()

    def split_and_mse(index, value, mask_val):
        actual_result = lambda: (
            mse_calculator(*test_split(index, value, mask, train), train[:, -1]),
            index,
            value
        )
        placeholder_result = lambda: (np.inf, -1, -1.0) # if 'value' is from sample not in current mask
        return cond(
            pred=mask_val, 
            true_fun=actual_result, 
            false_fun=placeholder_result
        )

    lowest_mse = np.inf
    b_i = None
    b_v = None
    group = train[mask]
    for sample in group:
        for _i, _v in enumerate(sample[:-1]):
            new_mask = group[:, _i] < _v
            left = group[new_mask]
            right = group[~new_mask]
            mse = mse_calculator_nondiff(left, right)
            if mse<lowest_mse:
                lowest_mse = mse
                b_i = _i
                b_v = _v
                
    left_mask, right_mask = test_split(b_i, b_v, mask, train)
    
    return SplitInfo(
        index=np.int8(b_i),
        value=np.float32(b_v),
        left_mask=left_mask,
        right_mask=right_mask
    )
        
    # mses, indices, values = jax.vmap(split_and_mse)(flat_indices, flat_values, flat_mask)
    mses, indices, values = np.array([split_and_mse(i, v, m) for i, v, m in zip(flat_indices, flat_values, flat_mask)]).T
    indices = indices.astype(np.int16)
    # Find the best split
    best_idx = np.argmin(mses)
    best_index = indices[best_idx]
    best_value = values[best_idx]
    
    return best_index, best_value, test_split(best_index, best_value, mask, train)


def mse_calculator_nondiff(left, right) :
        def mse(samples) :
            outcomes = samples[:, -1]
            if len(outcomes) == 0:
                return 0
            
            mean_out = np.sum(outcomes) / len(outcomes)
            return np.sum((outcomes - mean_out) ** 2) / len(outcomes)
        
        left_mse = mse(left)
        right_mse = mse(right)
        #This represents the MSE of the split
        mse_split = (left_mse * len(left) + right_mse * len(right)) / (len(left) + len(right))

        return mse_split


def mse_calculator(left_mask, right_mask, outcomes):
    def group_loss(mask):
        if np.sum(mask) == 0:
            return 0
        # Compute mean outcome of all samples in current group
        mean_outcome = diffable_subset_mean(outcomes, mask)
        
        squared_errors = np.where(mask, (outcomes - mean_outcome)**2, 0.0)
        weighted_mse = np.sum(squared_errors) # ("weighted" means multiply with group size. Hence mse -> se)
        return weighted_mse

    mse_split = (group_loss(left_mask) + group_loss(right_mask)) / (np.sum(left_mask) + np.sum(right_mask))
    return mse_split


def test_split(index, value, mask, train):
    # determine splitting masks and apply mask of current train subset (logical and)
    global_left_mask = train[:, index] < value
    return np.logical_and(global_left_mask, mask), np.logical_and(~global_left_mask, mask)
