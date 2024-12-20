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
    global pruned_nodes, explored_nodes

    # Draw node
    if depth == 0:
        # Terminal node: Assign random value
        if node_id not in st.session_state.terminal_values:
            # Random values for terminal nodes between -10 and 10
            st.session_state.terminal_values[node_id] = random.randint(-10, 10)

        ax.text(x, y, f"{st.session_state.terminal_values[node_id]}", fontsize=10, ha='center', va='center', bbox=dict(boxstyle="circle", facecolor="white"))
        explored_nodes.append(node_id)  # Mark as explored
        return st.session_state.terminal_values[node_id], alpha, beta

    player = "Max" if is_max else "Min"
    node_color = "lightgreen" if node_id not in pruned_nodes else "lightcoral"  # Green for explored, Red for pruned

    ax.text(x, y, player, fontsize=10, ha='center', va='center', bbox=dict(boxstyle="circle", facecolor=node_color))

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

    # Highlight current node value
    node_text_color = "blue" if node_id == "Root" else "red"
    ax.text(x, y - 10, f"{node_value}", fontsize=8, ha='center', color=node_text_color)

    # Connect parent to children
    if parent_pos:
        conn = ConnectionPatch(parent_pos, (x, y), "data", "data", arrowstyle='->', color="black")
        ax.add_artist(conn)

    return node_value, alpha, beta

# Alpha-Beta Pruning Function
def alpha_beta_pruning(left_value, right_value, is_max, alpha, beta, node_id):
    global pruned_nodes

    if is_max:
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
    explored_nodes = []  # List to keep track of explored nodes
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

    # Highlight pruned nodes with a cross
    for pruned_node in pruned_nodes:
        pruned_node_id = pruned_node.split(":")[1].strip()
        node_pos = get_node_position(pruned_node_id)  # Get the position of the pruned node
        if node_pos:
            ax.text(node_pos[0], node_pos[1], "X", fontsize=12, ha='center', va='center', color="red")  # Mark with a red "X"

    # Display tree
    st.pyplot(fig)

    # Display Best Value
    st.sidebar.write(f"Best Value (Algorithm: {algorithm}): {best_value}")
    st.sidebar.write(f"Pruned Nodes: {', '.join(pruned_nodes)}")
    st.sidebar.write(f"Explored Nodes: {', '.join(explored_nodes)}")

else:
    st.session_state.tree_generated = False  # Reset flag when the button is not pressed

# Helper function to get node position for pruned nodes (you may need to adjust this depending on your tree structure)
def get_node_position(node_id):
    # This function should return the x, y coordinates of the node with node_id
    # For simplicity, just returning a random position for now (replace with actual logic)
    node_positions = {
        "RootL": (0.3, 0.7),
        "RootR": (0.7, 0.7),
        "RootLL": (0.2, 0.5),
        "RootLR": (0.4, 0.5),
        "RootRL": (0.6, 0.5),
        "RootRR": (0.8, 0.5),
    }
    return node_positions.get(node_id)
