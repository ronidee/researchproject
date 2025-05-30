import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_tree_splits(node, x_bounds=(0, 1), y_bounds=(0, 1), color='black', ax=None):
    """
    Recursively plot the decision boundaries of a decision tree on a 2D plane.
    Parameters:
        node (dict): Tree node.
        x_bounds (tuple): Min and max bounds for feature 0 (x-axis).
        y_bounds (tuple): Min and max bounds for feature 1 (y-axis).
        ax (matplotlib.axes.Axes): The axis to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

        minor_ticks = np.linspace(0, 1, 11)   # twice as many: 0.0, 0.125, ..., 1.0

        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        ax.set_aspect('equal')
        ax.grid(True, which="both", axis="both", color='lightgray', linestyle='--', linewidth=0.5)

    # Leaf node: nothing to split
    if 'label' in node:
        return

    index = node['index']
    value = node['value']

    # Vertical split (feature 0)
    if index == 0:
        ax.plot([value, value], y_bounds, color=color, linewidth=1.5)
        # Left: x < value
        plot_tree_splits(node['left'], (x_bounds[0], value), y_bounds, color=color, ax=ax)
        # Right: x >= value
        plot_tree_splits(node['right'], (value, x_bounds[1]), y_bounds, color=color, ax=ax)

    # Horizontal split (feature 1)
    elif index == 1:
        ax.plot(x_bounds, [value, value], color=color, linewidth=1.5)
        # Left: y < value
        plot_tree_splits(node['left'], x_bounds, (y_bounds[0], value), color=color, ax=ax)
        # Right: y >= value
        plot_tree_splits(node['right'], x_bounds, (value, y_bounds[1]), color=color, ax=ax)

    return ax

# Example usage:
if __name__ == "__main__":
    tree1 = {
        'index': 0, 'value': 0.6,
        'left': {
            'index': 1, 'value': 0.4,
            'left': {'label': 0},
            'right': {'label': 1}
        },
        'right': {
            'index': 1, 'value': 0.7,
            'left': {'label': 2},
            'right': {'label': 3}
        }
    }
    tree2 = {
        'index': 1, 'value': 0.8,
        'left': {
            'index': 0, 'value': 0.3,
            'left': {'label': 2},
            'right': {
                'index': 1, 'value': 0.2,
                'left': {'label': 2},
                'right': {'label': 3}
            }
        },
        'right': {'label': 2}
    }
    for tree, color in ((tree1, "red"), (tree2, "blue")):
        ax = plot_tree_splits(tree, x_bounds=(0, 1), y_bounds=(0, 1), color=color)
        plt.tight_layout()
    plt.show()
