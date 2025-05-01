import jax.numpy as jnp
import jax

import iteratively

def compute_tree_difference(client_tree, dummy_tree, client_y, dummy_y, initial_bounds):
    partitions, mask = get_intersected_partitions(client_tree, dummy_tree, initial_bounds)
    # partitions = jnp.unique(partitions, axis=0)
    partition_centers = jnp.array([partition_center(partition) for partition in partitions])
    
    client_predict_fn = lambda x: diffable_predict(client_tree, x, client_y)
    dummy_predict_fn = lambda x: diffable_predict(dummy_tree, x, dummy_y)
    
    fn_squared_errors = jax.vmap(lambda x: (client_predict_fn(x) - dummy_predict_fn(x))**2)
    squared_errors = fn_squared_errors(partition_centers)
    mse = jnp.sum(jnp.where(mask, squared_errors, 0.0)) / jnp.sum(mask)
    return mse

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

def extract_partitions(tree, initial_bounds):
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

def intersect_partitions(r1, r2):
    """Compute intersection of two axis-aligned hyperrectangles."""
    lower = jnp.maximum(r1[:, 0], r2[:, 0])
    upper = jnp.minimum(r1[:, 1], r2[:, 1])
    
    is_not_empty = jnp.all(lower < upper)
    intersection = jnp.stack([lower, upper], axis=-1)
    
    return intersection, is_not_empty

def partition_center(partition):
    assert 2 <= partition.ndim <= 3 # only allow single cell or array of cells

    # replace (-inf, +inf) rows with (-1, -1) for safe mean computation
    # value is irrelevant, as (-inf, +inf) means .predict() doesn't use the feature
    unsafe_mask = (partition[:, 0] == -jnp.inf) & (partition[:, 1] == jnp.inf)
    safe_partition = jnp.copy(partition)
    safe_partition = safe_partition.at[unsafe_mask].set([-1, -1])
    
    return jnp.mean(safe_partition, axis=partition.ndim-1)

def diffable_predict(node, sample, y_train, steepness=100.0):
    """
    Recursively computes a soft prediction.
    Instead of a hard branch, we use the sigmoid to softly weight left/right predictions.
    """
    # Base case: if the node is a terminal, return its value.
    if iteratively.is_leaf(node):
        return iteratively.to_leaf(node, y_train)
    
    # Get the split parameters.
    index = node['index']
    threshold = node['value']
    
    # Compute soft decision weight for left branch (close to 1 if sample[index] < threshold)
    left_weight = soft_step(sample[index], threshold, steepness)
    # The right branch gets the complementary weight.
    right_weight = 1.0 - left_weight
    
    # Recursively compute the predictions for left and right branches.
    left_pred = diffable_predict(node['left'], sample, y_train, steepness)
    right_pred = diffable_predict(node['right'], sample, y_train, steepness)
    
    # Return a weighted average.
    return left_weight * left_pred + right_weight * right_pred

def soft_step(x, threshold, steepness=100.0):
    # Approximates the indicator function x < threshold using a sigmoid.
    return jax.nn.sigmoid(steepness * (threshold - x))
