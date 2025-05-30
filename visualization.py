from turtle import st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _plot_potential_candidates(x, y, std, total, show):
    y1 = y - np.minimum(std, y)
    y2 = y + std

    plt.figure()
    plt.errorbar(x, y, yerr=[np.minimum(std, y), std], fmt='-o', capsize=5)
    # plt.plot(x, y)
    # plt.fill_between(x, y1, y2, alpha=0.1)
    plt.xlabel('Epochs')
    plt.ylabel(f'N.o. Potential Candidates (out of {total})')
    plt.title('Mean ± Std of Potential Candidates vs. Epochs')
    plt.grid(True, which='both', color='lightgray', linestyle='--')
    
    if show:
        plt.show()
    else:
        plt.savefig(f"out/figures/potential-candidates.png")

def _plot_feature_coverage(x, df, show):
    plt.figure()
    for i in range(8):
        mean_col = f'feature_{i}_percent_mean'
        std_col = f'feature_{i}_percent_std'
        plt.errorbar(x, df[mean_col], yerr=df[std_col],
                     fmt='-o', capsize=4, label=f'Feature {i}')

    plt.xlabel('Epochs')
    plt.ylabel('within bounds (%)')
    plt.title('Mean ± Std of Feature Coverage vs. Epochs')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig(f"out/figures/feature-coverage-errbars.png")

    plt.figure()
    for i in range(8):
        mean_col = f'feature_{i}_percent_mean'
        std_col = f'feature_{i}_percent_std'
        y = df[mean_col]
        std = df[std_col]

        plt.plot(x, y, label=f'Feature {i}')
        plt.fill_between(x, y-std, y+std, alpha=0.1)

    plt.xlabel('Epochs')
    plt.ylabel('within bounds (%)')
    plt.title('Mean ± Std of Feature Coverage vs. Epochs')
    plt.legend()
    plt.grid(True)

    if show:
        plt.show()
    else:
        plt.savefig(f"out/figures/feature-coverage-shaded.png")


def plot_evaluation(df, test_size, show):
    # 1. plot the number of potential candidates (lower is better) against the number of epochs
    x = df['epochs']
    # _plot_potential_candidates(x, df['candidates_mean'], df['candidates_std'], test_size, show)
    

    # 2. plot the feature range covered by the bounds (lower is better) against the number of epochs
    _plot_feature_coverage(x, df, show)


def draw_tree_partition(show):
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

    xlim=(0, 100)
    ylim=(0, 100)
    resolution=400

    """Visualizes the decision tree's partitioning in a 2D input space."""
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    preds = np.array([predict(tree, point) for point in grid])
    preds = preds.reshape(xx.shape)
    colors = {0: "#FFDAB9", 1: "#ADD8E6"}  # light orange, light blue

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, preds, levels=[-0.1, 0.5, 1.1], colors=[colors[0], colors[1]])

    legend_elements = [
        Patch(facecolor=colors[1], label="yes"),
        Patch(facecolor=colors[0], label="no")
    ]
    plt.legend(handles=legend_elements)
    plt.xlabel("price (€)")
    plt.ylabel("size (cm)")
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("out/figures/tree-partition.png")

def plot_splits(show):
    def _plot_splits(node, x_bounds=(0, 1), y_bounds=(0, 1), color='black', ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

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
            _plot_splits(node['left'], (x_bounds[0], value), y_bounds, color=color, ax=ax)
            # Right: x >= value
            _plot_splits(node['right'], (value, x_bounds[1]), y_bounds, color=color, ax=ax)

        # Horizontal split (feature 1)
        elif index == 1:
            ax.plot(x_bounds, [value, value], color=color, linewidth=1.5)
            # Left: y < value
            _plot_splits(node['left'], x_bounds, (y_bounds[0], value), color=color, ax=ax)
            # Right: y >= value
            _plot_splits(node['right'], x_bounds, (value, y_bounds[1]), color=color, ax=ax)

        return ax

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
    
    trees = [tree1, tree2]
    colors = ["red", "blue"]

    for i in range(2):
        _plot_splits(trees[i], x_bounds=(0, 1), y_bounds=(0, 1), color=colors[i])
        plt.tight_layout()
        if not show:
            plt.savefig(f"out/figures/tree-splits-{i}.png")

    if show:
        plt.show()
            

