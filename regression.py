import json
import numpy as np
from random import randrange
import random

import utils

# Calculate accuracy percentage
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def test_tree(tree, test_data):
    predictions = tree.predict(test_data[:, :-1])
    
    return {
        'mse': mse(test_data[:, -1], predictions),
        'mae': mae(test_data[:, -1], predictions),
        'predictions': predictions
    }


# based on the fantastic implementation from Maha Chakir at https://medium.com/@mahachakir/exploring-regression-trees-building-from-scratch-in-python-9d25dd0dc6ca
class RegressionTree:
    def __init__(self, max_depth=None, min_size=None, root=None, fingerprint=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = root
        self.final_depth = 0

        if root:
            assert fingerprint # loading trained tree requires original fingerprint
            self.fingerprint = fingerprint

    # Wrapper for non-instance __predict() function
    def predict(self, X):
        if X.ndim == 1:
            return self.__predict(self.root, X)
        elif X.ndim == 2:
            return np.array([self.__predict(self.root, sample) for sample in X])
            
    # Make a prediction with a decision tree
    def __predict(self, node, sample):
        if sample[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.__predict(node['left'], sample)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.__predict(node['right'], sample)
            else:
                return node['right']


    # Fit tree to 'train' data, which contains labels at last column
    def fit(self, train, retrain=False):
        if self.root and not retrain:
            raise AssertionError("Tree has already been trained. " \
                "If you're sure you want to fit it again, pass 'retrain=True'.")
        
        if not (self.max_depth and self.min_size):
            raise AssertionError("Attributes 'max_depth' and 'min_size' must be set before calling '.fit()'.")
        
        self.feature_bounds = utils.get_feature_bounds(train, extend_int_bounds=False)
        self.root = self.get_split(train)
        self.split(self.root, 1)

        # after tree is formed, create fingerprint
        self.fingerprint = self.generate_fingerprint(train)
    
    # split samples based on random feature and value
    def get_split(self, samples):
        # We use random split points, to get different trees using the same data
        feature_index = randrange(samples.shape[1] - 1)
        split_value = np.random.uniform(*self.feature_bounds[feature_index])
        groups = self.test_split(feature_index, split_value, samples)
        
        return {'index': feature_index,'value': split_value, 'groups': groups}
    
    
    def mse_calculator(self, left, right) :
        def _mse(samples) :
            outcomes = samples[:, -1]
            if len(outcomes) == 0:
                return 0
            
            mean_out = np.sum(outcomes) / len(outcomes)
            return np.sum((outcomes - mean_out) ** 2) / len(outcomes)
        
        left_mse = _mse(left)
        right_mse = _mse(right)
        #This represents the MSE of the split
        mse_split = (left_mse * len(left) + right_mse * len(right)) / (len(left) + len(right))

        return mse_split
    
    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, samples):
        mask = samples[:, index] < value
        return samples[mask], samples[~mask]

    # recursively split a node (until max depth or min size is reached)
    def split(self, node, depth):
        if depth > self.final_depth: self.final_depth = depth
        
        left, right = node.pop('groups')

        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_leaf(left), self.to_leaf(right)
            return

        # process left child
        node['left'] = self.to_leaf(left) if left.shape[0] <= self.min_size else self.get_split(left)
        if isinstance(node['left'], dict):
            groups = node['left']['groups']
            if groups[0].size == 0 or groups[1].size == 0:
                # detect a pseudo-split and prevent duplicate leaf nodes
                node['left'] = self.to_leaf(np.vstack(groups))
            else:
                # not a leaf node, not a pseudo-split: keep processing
                self.split(node['left'], depth + 1)

        # process right child
        node['right'] = self.to_leaf(right) if right.shape[0] <= self.min_size else self.get_split(right)
        if isinstance(node['right'], dict):
            groups = node['right']['groups']
            if groups[0].size == 0 or groups[1].size == 0:
                # detect a pseudo-split and prevent duplicate leaf nodes
                node['right'] = self.to_leaf(np.vstack(groups))
            else:
                # not a leaf node, not a pseudo-split: keep processing
                self.split(node['right'], depth + 1)

    # compute leaf value using labels of given samples (mean)
    def to_leaf(self, samples):
        return np.mean(samples[:, -1])


    # create fingerprint from train params and tree hash
    # trees with different fingerprints may still satisfy the .equals() function
    def generate_fingerprint(self, train):
        # create hash by hashing the concatenated dict and ds hashes
        root_hash = utils.smolhash(self)
        train_hash = utils.smolhash(train) # stop gradient during grad calculation
        tree_hash = utils.smolhash(root_hash + train_hash)
        
        return f"u{self.max_depth}-l{self.min_size}-{tree_hash}"

    def to_json(self):
        return json.dumps(
            obj={
                "root": self.root,
                "max_depth": self.max_depth,
                "min_size": self.min_size,
                "fingerprint": self.fingerprint
            },
            indent=2
        )
    
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)


