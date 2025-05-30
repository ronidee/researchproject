import json
import jax.numpy as jnp
import jax
from jax.lax import cond
import numpy as np
import random
from dataclasses import dataclass
from differentiable import diffable_predict
from jax.interpreters.ad import JVPTracer

_feature_bounds = None
random.seed(1)


disable_jax = False

class TreeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, JVPTracer):
            return o.primal
        elif (isinstance(o, (jax.numpy.ndarray, np.ndarray)) and o.size == 1) or isinstance(o, jnp.float32):
            return o.item()
        elif isinstance(o, jnp.int8):
            return o.item()
        elif isinstance(o, jnp.ndarray):
            return str(o)
        else:
            print(type(o), o)
            return o


if disable_jax:
    def non_diffable_cond(pred, true_fun, false_fun, *operand):
        if pred:
            return true_fun(*operand)
        else:
            return false_fun(*operand)

    cond = non_diffable_cond
    jnp = np


@dataclass
class SplitInfo:
    index: jax.Array
    value: jax.Array
    left_mask: jnp.ndarray
    right_mask: jnp.ndarray
    
    def __post_init__(self):        
        assert isinstance(self.index, jax.Array if not disable_jax else object), "TypeError for field 'index'"
        assert isinstance(self.value, jax.Array if not disable_jax else object), "TypeError for field 'value'"
        assert isinstance(self.left_mask, jnp.ndarray) and self.left_mask.dtype==bool , "TypeError for field 'left_mask'"
        assert isinstance(self.right_mask, jnp.ndarray) and self.right_mask.dtype==bool , "TypeError for field 'right_mask'"
        
        # self.is_empty_split = self.left_mask.any() and self.right_mask.any()

def mse(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))

def test_tree(tree, test_data):
    X, y = test_data[:, :-1], test_data[:, -1]

    # predictions = predict(tree, X)
    predictions = jnp.array([diffable_predict(tree, sample, tree['y_train']) for sample in X])
    return {
        'mse': mse(y, predictions),
        'mae': mae(y, predictions),
        'predictions': predictions
    }


def predict(root, X):
        if X.ndim == 1:
            return __predict(root, X, root['y_train'])
        elif X.ndim == 2:
            return jnp.array([__predict(root, sample, root['y_train']) for sample in X])


def __predict(node, sample, y_train):
    if is_leaf(node):
        return to_leaf(node, y_train)
    else:
        decision = 'left' if sample[node['index']] < node['value'] else 'right'
        return __predict(node[decision], sample, y_train)
    
def to_leaf(node, y_train):
    return diffable_subset_mean(y_train, node['subset_mask'])


def make_process_node_fn(train, max_depth, min_size, find_best_split):
    
    # --- all of these are pure JAX functions ---
    def split_node(node):
        info = find_best_split(node['subset_mask'], train)
        return {
            'subset_mask': node['subset_mask'],
            'index':       info.index,
            'value':       info.value,
            'left':        {'subset_mask': info.left_mask},
            'right':       {'subset_mask': info.right_mask},
        }
                
    def is_split_allowed(node, depth):
        return (node['subset_mask'].sum() > min_size) & (depth < max_depth)

    def equal_leafs(left, right):
        return to_leaf(left, train[:, -1]) == to_leaf(right, train[:, -1])

    def is_empty_split(node):
        lm, rm = node['left']['subset_mask'], node['right']['subset_mask']
        return ~(lm.any() & rm.any())


    def process_node(node, depth):
        # 1) split
        node2 = split_node(node)

        # 2) two cases: empty split → immediate leaf, else → examine children
        def on_empty(n):
            return n, False, False, True

        def on_not_empty(n):
            left, right = n['left'], n['right']

            # 3) decide for each child whether it stays raw (queue) or becomes a leaf
            ok_l = is_split_allowed(left, depth + 1)
            ok_r = is_split_allowed(right, depth + 1)
            
            # 5) maybe collapse into a single leaf if both children agree
            should_collapse = jnp.logical_and(jnp.logical_not(jnp.logical_or(ok_l, ok_r)), equal_leafs(left, right))

            return cond(should_collapse,
                        lambda: (n, ok_l, ok_r, True),
                        lambda: (n, ok_l, ok_r, False))

        return cond(is_empty_split(node2),
                    on_empty, on_not_empty,
                    node2)

    return process_node

def train_tree(train, max_depth, min_size):
    def set_in(tree, path, new_sub):
        if not path:
            return new_sub
        head, *tail = path
        child = tree[head]
        updated = set_in(child, tail, new_sub)
        return { **tree, head: updated }

    process_node = make_process_node_fn(train, max_depth, min_size, find_best_split)

    root = {'subset_mask': jnp.ones(train.shape[0], dtype=bool)}
    raw_nodes = [ ((), root) ]

    for depth in range(max_depth):
        new_queue = []
        for path, node in raw_nodes:
            # run all splitting + leaf‐logic in one JAX call
            node_out, left_ok, right_ok, should_collapse = process_node(node, depth)
            
            if should_collapse:
                node_out = {
                    'subset_mask': node_out['subset_mask'],
                    'index':       jnp.int8(-1),
                    'value':       jnp.float32(-1),
                    'left':        node_out['left'],
                    'right':       node_out['right']
                }

            root = set_in(root, path, node_out)
            
            # 2) update host‐side queue normally
            if bool(left_ok):
                new_queue.append((path + ('left',), node_out['left']))
            if bool(right_ok):
                new_queue.append((path + ('right',), node_out['right']))

        raw_nodes = new_queue

    # print(json.dumps(root, indent=2, cls=TreeEncoder))
    # print(train[:, -1])
    # exit()
    return root | {'y_train': train[:, -1]}


def is_leaf(node):
    return len(node) == 1


def diffable_subset_mean(arr, subset_mask):
    mean = jnp.sum(jnp.where(subset_mask, arr, 0.0)) / (jnp.sum(subset_mask) + 1e-14)
    return mean


def find_best_split(mask, train):    
    N, D = train[:, :-1].shape
    
    # Note: we want to try out *every* value for splitting at the current node (n.o. values=N*D)
    flat_indices = jnp.repeat(jnp.arange(D), N)  # shape: (D*N,)
    flat_values = train[:, :-1].T.flatten()  # shape: (D*N,)
    flat_mask = jnp.repeat(mask[:, None].T, D, axis=0).flatten()

    def split_and_mse(index, value, mask_val):
        actual_result = lambda: (
            mse_calculator(*test_split(index, value, mask, train), train[:, -1]),
            index,
            value
        )
        placeholder_result = lambda: (jnp.inf, -1, -1.0) # if 'value' is from sample not in current mask
        return cond(
            pred=mask_val, 
            true_fun=actual_result, 
            false_fun=placeholder_result
        )   
        
    # mses, indices, values = jax.vmap(split_and_mse)(flat_indices, flat_values, flat_mask)
    mses, indices, values = jnp.array([split_and_mse(i, v, m) for i, v, m in zip(flat_indices, flat_values, flat_mask)]).T
    indices = indices.astype(jnp.int16)
   
    # Find the best split
    best_idx = jnp.argmin(mses)
    best_index = indices[best_idx]
    best_value = values[best_idx]
    
    left_mask, right_mask = test_split(best_index, best_value, mask, train)#
    split_info = SplitInfo(
        index=jnp.int8(best_index),
        value=jnp.float32(best_value),
        left_mask=left_mask,
        right_mask=right_mask
    )
    
    return split_info


def mse_calculator(left_mask, right_mask, y_train):
    def group_loss(mask):
        # Compute mean outcome of all samples in current group
        mean_outcome = diffable_subset_mean(y_train, mask)
        squared_errors = jnp.where(mask, (y_train - mean_outcome)**2, 0.0)
        weighted_mse = jnp.sum(squared_errors) # ("weighted" means multiply with group size. Hence mse -> se)
        return weighted_mse

    mse_split = (group_loss(left_mask) + group_loss(right_mask)) / (jnp.sum(left_mask) + jnp.sum(right_mask) + 1e-14)
    return mse_split


def test_split(index, value, mask, train):
    # determine splitting masks and apply mask of current train subset (logical and)
    global_left_mask = train[:, index] < value
    return jnp.logical_and(global_left_mask, mask), jnp.logical_and(~global_left_mask, mask)
