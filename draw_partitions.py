import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Example decision tree dictionary
# Structure:
# - Internal node: {"feature": int, "value": float, "left": ..., "right": ...}
# - Leaf node: {"label": 0 or 1}

tree = {
    "feature": 0,
    "value": 5,
    "left": {"label": 0},
    "right": {
        "feature": 0,
        "value": 90,
        "left": {
            "feature": 0,
            "value": 90,
            "left": {
                "feature": 1,
                "value": 60,
                "left": {"label": 0},
                "right": {"label": 1}
            },
            "right": {"label": 0}
        },
        "right": {"label": 0}
    },
}

def predict(tree, x):
    """Predict label for a single input x using the tree."""
    while "label" not in tree:
        feature = tree["feature"]
        value = tree["value"]
        if x[feature] <= value:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree["label"]

def visualize_tree_partitions(tree, xlim=(0, 100), ylim=(0, 100), resolution=400):
    """Visualizes the decision tree's partitioning in a 2D input space."""
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict labels for all points in the grid
    preds = np.array([predict(tree, point) for point in grid])
    preds = preds.reshape(xx.shape)

    # Define colors
    colors = {0: "#FFDAB9", 1: "#ADD8E6"}  # light orange, light blue

    # Plot the partitions
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, preds, levels=[-0.1, 0.5, 1.1], colors=[colors[0], colors[1]])

    # Add legend
    legend_elements = [
        Patch(facecolor=colors[1], label="yes"),
        Patch(facecolor=colors[0], label="no")
    ]
    plt.legend(handles=legend_elements)

    plt.xlabel("price (€)")
    plt.ylabel("size (cm)")
    plt.grid(True)
    plt.show()

# Visualize the example tree
visualize_tree_partitions(tree)
