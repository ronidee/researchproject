import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters.ad import JVPTracer

from utils import smolhash


# This isn't actually large. But large in the context of tree diff errors (at least I hope so...)
VERY_LARGE_NUMBER = 10**0

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def test_tree(tree, data):
    predictions = [tree.predict(sample) for sample in data]
    return accuracy_metric(data[:, -1], predictions)


class DiffableTree:
    def __init__(self, max_depth=None, min_size=None, root=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = root
    
    
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
        assert type(train) == list
        
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
        class_values = list(jnp.unique(jnp.array(list(row[-1] for row in dataset))))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}


    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    
    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth+1)


    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(jnp.unique(jnp.array(outcomes)).tolist(), key=outcomes.count)


    # TODO: check if diffrent split values have same gini index
    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini
    
    
    # Check whether this tree instance is equal to tree instance 'tree2'
    def equals(self, tree2):
        return tree_diff(self, tree2) == 0

    
    # create fingerprint from train params and tree hash
    # trees with different fingerprints may still satisfy the .equals() function
    def generate_fingerprint(self, train):
        # create hash by hashing the concatenated dict and ds hashes
        root_hash = smolhash(self)
        train_hash = smolhash(np.array(jax.lax.stop_gradient(train))) # stop gradient during grad calculation
        tree_hash = smolhash(root_hash + train_hash)
        
        return f"u{self.max_depth}-l{self.min_size}-{tree_hash}"


# computes diff between two trees by summed squared distance of indexes/thresholds.
# Different structure (leaf vs non-leaf) inflicts instant 'VERY_LARGE_NUMBER' damage
# @param level: level in the tree hierachy (root node=0), so we know when to sum up. Could replace by always summing up. TODO!
# TODO: remove dependence on same random seed during training
def tree_diff(tree1, tree2, client_test, level=0):
    preds_1 = [tree1.predict(sample) for sample in client_test]
    preds_2 = [tree2.predict(sample) for sample in client_test]

    print(preds_1, "\n", preds_2)
    print(np.sum(np.array(preds_1) - np.array(preds_2)))
    exit()

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
            errors.append(jnp.pow(v1-v2, 2))
    
    # return sum of aggregated errors
    return jnp.sum(jnp.array(errors)) if level == 0 else errors

    
# DANGEROUS: expects only 'b' to ever be traced by JAX
def same_type(a, b):
    # Check if a, b have same type or, if not and b is wrapped 
    # in a Tracer object, if a and b's wrapped variable have same type
    return type(a) == type(b) \
        or (isinstance(b, jax._src.interpreters.ad.JVPTracer) and type(a) == type(b.primal))
