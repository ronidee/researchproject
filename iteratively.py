import numpy as np
from random import randrange
import random


__feature_bounds = None

def train_tree(train, max_depth=3, min_size=3, feature_bounds=None):
    global __feature_bounds
    __feature_bounds = feature_bounds
    random.seed(1)
    root = {"samples": np.ones(train.shape[0], dtype=bool)}
    latest_nodes = [root]
    
    depth = 1
    while depth < max_depth: # Note: the very last round (all to leaf) is completed after the loop
        for node in latest_nodes:
            new_nodes = []
            process_node(node, min_size, train)
            
            # leaf: single value array. non-leaf node: dict
            if not isinstance(node['left'], np.float64):
                new_nodes.append(node['left'])
            if not isinstance(node['right'], np.float64):
                new_nodes.append(node['right'])
            
        latest_nodes = new_nodes
        depth += 1
    
    # last round, convert all non-leaf nodes to leafs
    for node in latest_nodes:
        split(node, train)
        node['left'] = to_leaf(node['left']['samples'], train[:, -1])
        node['right'] = to_leaf(node['right']['samples'], train[:, -1])

    return root


def process_node(node, min_size, train):
    # split subset of current node into left and right child (sub-) subset    
    # split deletes keys ['samples'] and adds ['left', 'right', 'index', 'value']
    split(node, train)

    left_mask = node['left']['samples']
    right_mask =  node['right']['samples']
    
    # Turn newly created child-nodes to leafs if they're too small
    if np.sum(left_mask) < min_size:
        node['left'] = to_leaf(left_mask, train[:, -1])
    
    if np.sum(right_mask) < min_size:
        node['right'] = to_leaf(right_mask, train[:, -1])
        

def to_leaf(samples_mask, train):
    return diffable_subset_mean(train, samples_mask)


def diffable_subset_mean(arr, subset_mask):
    mean = np.sum(np.where(subset_mask, arr, 0.0)) / (np.sum(subset_mask) + 1e-8)
    return mean


def split(node, train):
    subset_mask = node.pop('samples')

    feature_index = randrange(train.shape[1] - 1)
    split_value = np.random.uniform(*__feature_bounds[feature_index])
    left_mask, right_mask = test_split(feature_index, split_value, subset_mask, train)
    
    # apply changes to node (in-place)
    node['left'] = {'samples': left_mask}
    node['right'] = {'samples': right_mask}
    node['index'] = feature_index
    node['value'] = split_value
    

def test_split(index, value, mask, train):
    # determine splitting masks and apply mask of current train subset (logical and)
    left_mask = train[:, index] < value
    return np.logical_and(left_mask, mask), np.logical_and(~left_mask, mask)
