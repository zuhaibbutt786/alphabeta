import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import random  # For generating random values

# Streamlit App Configuration
st.set_page_config(page_title="Decision Tree with Minimax", layout="wide")
st.title("Decision Tree with Minimax and Alpha-Beta Pruning")

# Input Fields
st.sidebar.header("Tree Configuration")
depth = st.sidebar.number_input("Enter Tree Depth:", min_value=1, max_value=10, step=1)
algorithm = st.sidebar.radio("Select Algorithm:", ("Minimax", "Alpha-Beta Pruning"))

# Initialize session state variables
if "terminal_values" not in st.session_state:
    st.session_state.terminal_values = {}

if "tree_generated" not in st.session_state:
    st.session_state.tree_generated = False

# Recursive Tree Drawing Function
def draw_tree(ax, depth, x, y, step_x, step_y, is_max, node_id, parent_pos, alpha, beta):
    global pruned_nodes

    # Draw node
    if depth == 0:
        # Terminal node: Assign random value
        if node_id not in st.session_state.terminal_values:
            # Random values for terminal nodes between -10 and 10
            st.session_state.terminal_values[node_id] = random.randint(-10, 10)

        ax.text(x, y, f"{st.session_state.terminal_values[node_id]}", fontsize=10, ha='center', va='center', bbox=dict(boxstyle="circle", facecolor="white"))
        return st.session_state.terminal_values[node_id], alpha, beta

    player = "Max" if is_max else "Min"
    ax.text(x, y, player, fontsize=10, ha='center', va='center', bbox=dict(boxstyle="circle", facecolor="white"))

    # Create children positions
    left_x = x - step_x / 2
    right_x = x + step_x / 2
    child_y = y - step_y

    # Draw child nodes recursively
    left_value, alpha, beta = draw_tree(ax, depth - 1, left_x, child_y, step_x / 2, step_y, not is_max, f"{node_id}L", (x, y), alpha, beta)
    right_value, alpha, beta = draw_tree(ax, depth - 1, right_x, child_y, step_x / 2, step_y, not is_max, f"{node_id}R", (x, y), alpha, beta)

    # Minimax/Alpha-Beta logic
    if algorithm == "Minimax":
        node_value = max(left_value, right_value) if is_max else min(left_value, right_value)
    else:
        node_value, alpha, beta = alpha_beta_pruning(left_value, right_value, is_max, alpha, beta, node_id)

    ax.text(x, y - 10, f"{node_value}", fontsize=8, ha='center', color="red")

    # Connect parent to children
    if parent_pos:
        conn = ConnectionPatch(parent_pos, (x, y), "data", "data", arrowstyle='->', color="black")
        ax.add_artist(conn)

    return node_value, alpha, beta

# Alpha-Beta Pruning Function
def alpha_beta_pruning(left_value, right_value, is_max, alpha, beta, node_id):
    if is_max:
        # Max node
        if left_value > beta:
            # Prune right child
            pruned_nodes.append(f"Pruned Right: {node_id}")
            return left_value, alpha, beta  # Pruned branch
        alpha = max(alpha, left_value)
        if right_value > beta:
            # Prune the right branch
            pruned_nodes.append(f"Pruned Right: {node_id}")
            return left_value, alpha, beta
        return max(left_value, right_value), alpha, beta
    else:
        # Min node
        if right_value < alpha:
            # Prune left child
            pruned_nodes.append(f"Pruned Left: {node_id}")
            return right_value, alpha, beta  # Pruned branch
        beta = min(beta, right_value)
        if left_value < alpha:
            # Prune the left branch
            pruned_nodes.append(f"Pruned Left: {node_id}")
            return right_value, alpha, beta
        return min(left_value, right_value), alpha, beta


# Generate Tree Button
if st.sidebar.button("Generate Tree"):
    st.session_state.tree_generated = True  # Set flag to indicate tree generation has been initiated
    pruned_nodes = []  # List to keep track of pruned nodes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Root node position
    root_x = 0.5
    root_y = 0.9
    step_x = 0.4
    step_y = 0.1

    # Draw the tree
    best_value, alpha, beta = draw_tree(ax, depth, root_x, root_y, step_x, step_y, True, "Root", None, -float('inf'), float('inf'))

    # Display tree
    st.pyplot(fig)

    # Display Best Value
    st.sidebar.write(f"Best Value (Algorithm: {algorithm}): {best_value}")
    st.sidebar.write(f"Pruned Nodes: {', '.join(pruned_nodes)}")

else:
    st.session_state.tree_generated = False  # Reset flag when the button is not pressed
