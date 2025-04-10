import json
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import graphviz
from hashlib import md5

import differentiable





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
        something = json.dumps(something.root, cls=differentiable.JaxTracerEncoder).encode('utf-8')
    elif not isinstance(something, bytes):
        raise TypeError("Argument 'something' must be str, int, float, bytes, numpy.ndarray, jax.numpy.ndarray or differentiable.DiffableTree")
    
    return md5(something).hexdigest()[:7]

def get_ok_to_write_file(fp, create_dir=False, ignore_exist=False):
    if not ignore_exist and fp.exists():
        user_input = input(f"File already exists! Overwrite? ({fp.absolute().as_posix()}) [y/N]: ")
        return user_input.lower() == 'y'
    elif fp.parent.exists():
        return True
    elif create_dir or input(f"Directory doesn't exist. Create? ({fp.parent.absolute().as_posix()}) [y/N]: ").lower() == 'y':
        fp.parent.mkdir(parents=True)
        return True
    
    return False

def load_client_state(state_dir):
    # load client tree (json) and train dataset (npy)
    tree_data=(state_dir / "client_tree.json").read_text()
    client_tree = differentiable.DiffableTree.from_json(tree_data)
    client_train = np.load((state_dir / "client_train.npy")).astype(np.float32)

    # Build DiffableTree instance from tree dict and return client dataset, tree
    return client_train, client_tree

def save_client_state(state_dir, client_train, client_tree):
    # Create file pointers for saving ds and tree 
    fp_client_train = state_dir / "client_train.npy"
    fp_client_tree = state_dir / "client_tree.json"
    
    # save tree as json, if file doesn't exist or user doesn't care
    if get_ok_to_write_file(fp_client_tree):
        fp_client_tree.write_text(client_tree.to_json())
    
    # save train data as npy, if file doesn't exist or user doesn't care
    if get_ok_to_write_file(fp_client_train):
        jnp.save(fp_client_train, client_train)

def load_dummy_state(state_dir):
    # load dummy tree (json) and attack sample (npy)
    fp_tree = state_dir / "dummy_tree.json"
    fp_sample = state_dir / "dummy_sample.npy"
    
    dummy_sample = dummy_tree = None
    
    if fp_tree.exists():
        tree_data = (state_dir / "dummy_tree.json").read_text()
        dummy_tree = differentiable.DiffableTree.from_json(tree_data)
    
    if fp_sample.exists():
        dummy_sample = jnp.load(fp_sample).astype(np.float32)
    
    # Build DiffableTree instance from tree dict and return dummy sample, tree
    return dummy_sample, dummy_tree

def save_dummy_state(state_dir, dummy_sample=None, dummy_tree=None, ignore_conflicts=False):
    # File pointer for saving dummy train data
    if dummy_sample != None:
        fp_dummy_sample = state_dir / "dummy_sample.npy"
        if get_ok_to_write_file(fp_dummy_sample, create_dir=ignore_conflicts, ignore_exist=ignore_conflicts):
            jnp.save(fp_dummy_sample, dummy_sample)
    
    if dummy_tree != None:
        fp_dummy_tree = state_dir / "dummy_tree.json"
        if get_ok_to_write_file(fp_dummy_tree, create_dir=ignore_conflicts, ignore_exist=ignore_conflicts):
            fp_dummy_tree.write_text(dummy_tree.to_json())

# Visualize and render the tree.
def visualize_tree(client_tree, dummy_tree=None, fp_out=None, view=False):
    dot = graphviz.Digraph()
    node_id_counter = [0]  # mutable counter for unique node IDs

    def add_node(tree, parent_id=None, edge_label="", subgraph=None):
        """ Recursively adds nodes to the Graphviz object. """
        node_id = str(node_id_counter[0])
        node_id_counter[0] += 1

        label = f"feature: {tree['index']}\n{tree['value']}"
        subgraph.node(node_id, label=label)

        if parent_id is not None:
            subgraph.edge(parent_id, node_id, label=edge_label)

        for side in ("left", "right"):
            child = tree[side]
            if isinstance(child, dict):
                add_node(child, parent_id=node_id, edge_label=side, subgraph=subgraph)
            else:
                leaf_id = str(node_id_counter[0])
                node_id_counter[0] += 1
                subgraph.node(leaf_id, label=str(child), shape="box")
                subgraph.edge(node_id, leaf_id, label=side)

    # Create a subgraph for the first tree
    with dot.subgraph(name="cluster_0") as sub1:
        sub1.attr(label="Client Tree")
        add_node(client_tree, subgraph=sub1)

    if dummy_tree:
        # Create a subgraph for the second tree
        with dot.subgraph(name="cluster_1") as sub2:
            sub2.attr(label="Dummy Tree")
            add_node(dummy_tree, subgraph=sub2)

    dot.render(fp_out, cleanup=True, view=view, format="png")
    print("Saved tree diagram at:", fp_out.with_suffix(".png").absolute().as_posix())
