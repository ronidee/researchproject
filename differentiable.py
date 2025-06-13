import jax.numpy as jnp
import jax

import iteratively

def compute_tree_difference(client_tree, dummy_tree, client_y, dummy_y, initial_bounds):
    # feature ranges not normalized yet. TODO: normalize features
    refined_partition, mask = create_intersected_partition(client_tree, dummy_tree, initial_bounds)
    cell_centers = jnp.array([cell_center(cell) for cell in refined_partition])
    cell_sizes = jnp.array([cell_size(cell) for cell in refined_partition])
    
    client_predict_fn = lambda x: diffable_predict(client_tree, x, client_y)
    dummy_predict_fn = lambda x: diffable_predict(dummy_tree, x, dummy_y)
    
    fn_squared_errors = jax.vmap(lambda x: (client_predict_fn(x) - dummy_predict_fn(x))**2)
    squared_errors = fn_squared_errors(cell_centers)
    mse = jnp.sum(jnp.where(mask, squared_errors, 0.0)) * cell_sizes / jnp.sum(mask)
    return mse

def cell_size(cell):
    return jnp.prod(jnp.abs(cell[:, 0] - cell[:, 1]))
    
def create_intersected_partition(client_tree, dummy_tree, initial_bounds):
    client_cells = jnp.array(extract_partition(client_tree, initial_bounds))
    dummy_cells = jnp.array(extract_partition(dummy_tree, initial_bounds))

    client_cells = jnp.unique(client_cells, axis=0)
    dummy_cells = jnp.unique(dummy_cells, axis=0)
    
    # Create all index pairs (i,j)
    client_cells_idx, dummy_cells_idx = jnp.meshgrid(jnp.arange(client_cells.shape[0]), jnp.arange(dummy_cells.shape[0]), indexing='ij')
    client_cells_idx = client_cells_idx.flatten()
    dummy_cells_idx = dummy_cells_idx.flatten()

    # Extract all pairs
    client_cells_flat = client_cells[client_cells_idx]
    dummy_cells_flat = dummy_cells[dummy_cells_idx]

    # Vectorize the intersection function over these pairs
    intersections, mask = jax.vmap(intersect_cells)(client_cells_flat, dummy_cells_flat)

    return intersections, mask

def extract_partition(tree, initial_bounds):
    """
    Recursively extract all decision cells from a RegressionTree.
    Each partitions is represented as a list of (lower, upper) bounds for each feature.
    """
    def recurse(node, bounds):
        index = node.get('index', None)
        value = node.get('value', None)

        # Base case: if this is a leaf node
        if iteratively.is_leaf(node):
            yield bounds
            return

        # If left child exists
        left_bounds = [b if i != index else (b[0], min(b[1], value)) for i, b in enumerate(bounds)]
        if not iteratively.is_leaf(node['left']):
            yield from recurse(node['left'], left_bounds)
        else:
            yield left_bounds

        # If right child exists
        right_bounds = [b if i != index else (max(b[0], value), b[1]) for i, b in enumerate(bounds)]
        if not iteratively.is_leaf(node['right']):
            yield from recurse(node['right'], right_bounds)
        else:
            yield right_bounds
    
    return list(recurse(tree, initial_bounds))

def intersect_cells(c1, c2):
    # Compute intersection of two axis-aligned hyperrectangles (here: cells of a partition)
    lower = jnp.maximum(c1[:, 0], c2[:, 0])
    upper = jnp.minimum(c1[:, 1], c2[:, 1])
    
    is_not_empty = jnp.all(lower < upper)
    intersection = jnp.stack([lower, upper], axis=-1)
    
    return intersection, is_not_empty

def cell_center(partition):
    assert 2 <= partition.ndim <= 3 # only allow single cell or array of cells

    # replace (-inf, +inf) rows with (-1, -1) for safe mean computation
    # value is irrelevant, as (-inf, +inf) means .predict() doesn't use the feature
    unsafe_mask = (partition[:, 0] == -jnp.inf) & (partition[:, 1] == jnp.inf)
    safe_partition = jnp.copy(partition)
    safe_partition = safe_partition.at[unsafe_mask].set([-1, -1])
    
    return jnp.mean(safe_partition, axis=partition.ndim-1)

def diffable_predict(node, sample, y_train, steepness=100.0):
    # recursively computes a soft prediction.
    # if the node is a terminal, return its value.
    if iteratively.is_leaf(node):
        return iteratively.to_leaf(node, y_train)
    
    index = node['index']
    threshold = node['value']
    
    # compute soft decision weight for left/right branch
    left_weight = soft_step(sample[index], threshold, steepness)
    right_weight = 1.0 - left_weight
    
    # compute predictions for both branches
    left_pred = diffable_predict(node['left'], sample, y_train, steepness)
    right_pred = diffable_predict(node['right'], sample, y_train, steepness)
    
    # Return a weighted average.
    return left_weight * left_pred + right_weight * right_pred

def soft_step(x, threshold, steepness=100.0):
    return jax.nn.sigmoid(steepness * (threshold - x))
