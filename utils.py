import json
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import graphviz
from hashlib import md5
from jax.interpreters.ad import JVPTracer

import differentiable


class JaxTracerEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, JVPTracer):
            return o.primal
        else:
            return o


# not a secure hash function. only use for fingerprints/filenames
# returns the first 7 chars form the hexencoded md5 hash of 'something'
def smolhash(something):
    if isinstance(something, str):
        something = something.encode('utf-8')
    elif isinstance(something, (int, float)):
        something = str(something).encode('utf-8')
    elif isinstance(something, np.ndarray):
        something = something.data.tobytes()
    elif isinstance(something, jnp.ndarray):
        something = np.array(something).data.tobytes()
    elif isinstance(something, differentiable.DiffableTree):
        something = json.dumps(something.root, cls=JaxTracerEncoder).encode('utf-8')
    elif not isinstance(something, bytes):
        raise TypeError("Argument 'something' must be str, int, float, bytes, numpy.ndarray, jax.numpy.ndarray or differentiable.DiffableTree")
    
    return md5(something).hexdigest()[:7]

def get_ok_to_write_file(fp):
    print(f"checking: {fp.parent.absolute().as_posix()}")
    if fp.exists():
        user_input = input(f"File already exists! Overwrite? ({fp.absolute().as_posix()}) [y/N]: ")
        return user_input.lower() == 'y'
    elif fp.parent.exists():
        return True
    else:
        user_input = input(f"Directory doesn't exist. Create? ({fp.parent.absolute().as_posix()}) [y/N]: ")
        if user_input.lower() == 'y':
            fp.parent.mkdir(parents=True)
            return True
    
    return False

def load_client_state(state_dir):
    # load client tree (json) and train dataset (npy)
    client_tree_root = json.loads((state_dir / "client_tree.json").read_text())
    client_train = jnp.load(state_dir / "client_train.npy")

    # Build DiffableTree instance from tree dict and return client dataset, tree
    return client_train, differentiable.DiffableTree(root=client_tree_root)

def save_client_state(state_dir, client_train, client_tree):
    # Create file pointers for saving ds and tree 
    fp_client_train = state_dir / "client_train.npy"
    fp_client_tree = state_dir / "client_tree.json"
    
    # save tree as json, if file doesn't exist or user doesn't care
    if get_ok_to_write_file(fp_client_tree):
        fp_client_tree.write_text(json.dumps(client_tree.root, indent=2))
    
    # save train data as npy, if file doesn't exist or user doesn't care
    if get_ok_to_write_file(fp_client_train):
        jnp.save(fp_client_train, client_train)

def load_dummy_state(state_dir):
    # load dummy tree (json) and attack sample (npy)
    dummy_tree_root = json.loads((state_dir / "dummy_tree.json").read_text())
    dummy_sample = jnp.load(state_dir / "dummy_sample.npy")
    
    # Build DiffableTree instance from tree dict and return dummy sample, tree
    return dummy_sample, differentiable.DiffableTree(root=dummy_tree_root)

def save_dummy_state(state_dir, dummy_sample=None, dummy_tree=None):
    # File pointer for saving dummy train data
    if dummy_sample != None:
        fp_dummy_sample = state_dir / "dummy_sample.npy"
        if get_ok_to_write_file(fp_dummy_sample):
            jnp.save(fp_dummy_sample, dummy_sample)
    
    if dummy_tree:
        fp_dummy_tree = state_dir / "dummy_tree.json"
        if get_ok_to_write_file(fp_dummy_tree):
            fp_dummy_tree.write_text(json.dumps(dummy_tree.root, indent=2, cls=JaxTracerEncoder))

# Visualize and render the tree.
def visualize_tree(tree, fp_out, view=False):
    dot = graphviz.Digraph()
    node_id_counter = [0]  # mutable counter

    def add_node(tree, parent_id=None, edge_label=""):
        # Create a unique id for the current node.
        node_id = str(node_id_counter[0])
        node_id_counter[0] += 1

        # Create a label that includes the index and value
        label = f"feature: {tree['index']}\n{tree['value']}"
        dot.node(node_id, label=label)

        # If there's a parent, add an edge
        if parent_id is not None:
            dot.edge(parent_id, node_id)

        # Process both child nodes
        for side in ("left", "right"):
            child = tree[side]
            if isinstance(child, dict):
                add_node(child, parent_id=node_id, edge_label=edge_label)
            else:
                # It's a leaf node, create a leaf!
                leaf_id = str(node_id_counter[0])
                node_id_counter[0] += 1
                dot.node(leaf_id, label=str(child))
                dot.edge(node_id, leaf_id, label=side)
    
    add_node(tree)
    dot.render(fp_out, cleanup=True, view=view, format='png')
