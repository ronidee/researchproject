import json
import jax.numpy as jnp
import numpy as np
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
    if fp.exists():
        user_input = input(f"File already exists! Overwrite? ({fp.absolute().as_posix()}) [y/N]: ")
        return user_input.lower() == 'y'
    else:
        return True

def load_client_state(state_dir):
    # load client tree stored as json file
    client_tree_root = json.loads((state_dir / "client_tree.json").read_text())
    # load client train dataset stored as *.npy file
    client_train = jnp.load(state_dir / "client_train.npy")

    # Build DiffableTree instance from tree dict and return client tree + dataset
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

def load_dummy_state(fp_dummy_train):
    return jnp.load(fp_dummy_train)

def save_dummy_state(fp_dummy_train, dummy_train):
    # File pointer for saving dummy train data
    
    # save dummy train data as npy, if file doesn't exist or user doesn't care
    if get_ok_to_write_file(fp_dummy_train):
        jnp.save(fp_dummy_train, dummy_train)
    