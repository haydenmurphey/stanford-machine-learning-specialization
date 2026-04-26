"""
Personal Lab Notes: Neural Networks & Decision Trees
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import logging

# ==============================================================================
# TOPIC 1: NEURAL NETWORKS (BINARY CLASSIFICATION)
# ==============================================================================

# Reduce TensorFlow verbosity for cleaner output
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

print("\n==================================================")
print("TOPIC 1: NEURAL NETWORKS")
print("==================================================")

# ==============================================================================
# DATA SETUP (Mock Data)
# ==============================================================================
# X shape: (1000, 400) - 1000 examples of 20x20 pixel images unrolled into 400 features.
# y shape: (1000, 1) - Labels: 0 (digit zero) or 1 (digit one).

print("\n--- Data Setup ---")
X_train_nn = np.random.rand(4, 400) 
y_train_nn = np.array([[0], [1], [0], [1]]) 

print('The shape of X_train is:', X_train_nn.shape)
print('The shape of y_train is:', y_train_nn.shape)

# ==============================================================================
# PART 1: TENSORFLOW / KERAS IMPLEMENTATION
# ==============================================================================
print("\n--- Part 1: TensorFlow Implementation ---")

# 1. Define the Neural Network Architecture
# Layer 1: 25 units, Layer 2: 15 units, Layer 3 (Output): 1 unit
model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    # Specify input size (400 features)
        Dense(25, activation='sigmoid', name='layer1'), 
        Dense(15, activation='sigmoid', name='layer2'), 
        Dense(1,  activation='sigmoid', name='layer3')  
    ], name="my_model" 
)

# View model architecture and parameter counts
model.summary()

# 2. Compile the model (Define Loss and Optimizer)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

# 3. Train the model
model.fit(X_train_nn, y_train_nn, epochs=5, verbose=0) # verbose=0 hides training output

# 4. Make Predictions
# Predict on the first example (needs to be reshaped to 2D array: 1 example, 400 features)
prediction = model.predict(X_train_nn[0].reshape(1,400))
print(f"Raw prediction probability: {prediction[0][0]:.4f}")

# Apply threshold
yhat = 1 if prediction >= 0.5 else 0
print(f"Prediction after threshold: {yhat}")


# ==============================================================================
# PART 2: NUMPY IMPLEMENTATION (Unvectorized)
# ==============================================================================
# z = w * x + b
# a = g(z)

def sigmoid(z):
    """Compute the sigmoid of z"""
    return 1 / (1 + np.exp(-z))

def my_dense(a_in, W, b, g):
    """
    Computes a single dense layer for ONE example.
    
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    : activation function (e.g. sigmoid)
    Returns:
      a_out (ndarray (j,))  : Output of the j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    
    for j in range(units):             
        w = W[:,j]                     # Get weights for unit j
        z = np.dot(w, a_in) + b[j]     # Dot product of inputs and weights + bias
        a_out[j] = g(z)                # Apply activation function
        
    return a_out

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    """Chains multiple dense layers together for ONE example."""
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return a3


# ==============================================================================
# PART 3: VECTORIZED NUMPY IMPLEMENTATION 
# ==============================================================================
# Instead of looping through units, use matrix multiplication (np.matmul) 
# to calculate all units and all examples simultaneously. Z = XW + b

def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer for MULTIPLE examples (Vectorized).
    
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    : activation function
    Returns:
      A_out (ndarray (m,j)) : Output for m examples, j units
    """
    Z = np.matmul(A_in, W) + b    
    A_out = g(Z)                 
    return A_out

def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    """Chains multiple vectorized layers together for MULTIPLE examples."""
    A1 = my_dense_v(X,  W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    A3 = my_dense_v(A2, W3, b3, sigmoid)
    return A3

print("\n--- Part 3: Vectorized NumPy Implementation Test ---")
# Extract trained weights from the TensorFlow model to test our NumPy functions
[layer1, layer2, layer3] = model.layers
W1_tmp, b1_tmp = layer1.get_weights()
W2_tmp, b2_tmp = layer2.get_weights()
W3_tmp, b3_tmp = layer3.get_weights()

# Make predictions on the entire dataset at once using NumPy matrix math
Predictions = my_sequential_v(X_train_nn, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)
Yhat = (Predictions >= 0.5).astype(int)

print(f"Vectorized predictions shape: {Predictions.shape}")
print(f"Predictions:\n{Yhat}")


# ==============================================================================
# PART 4: NUMPY BROADCASTING CONCEPTS
# ==============================================================================
print("\n--- Part 4: NumPy Broadcasting Examples ---")
# Broadcasting allows NumPy to perform element-wise operations on arrays of 
# different shapes by virtually "stretching" the smaller array.

# Example 1: Scalar to Vector
a = np.array([1, 2, 3]).reshape(-1,1)  # Shape (3,1)
b = 5                                  # Scalar
print(f"Shape of (a + b): {(a + b).shape}") 
# 'b' is broadcasted to (3,1) to match 'a'

# Example 2: Vector to Matrix (Used in our dense layer: Z = XW + b)
a = np.array([1, 2, 3, 4]).reshape(-1,1) # Shape (4,1) - 4 rows, 1 col
b = np.array([1, 2, 3]).reshape(1,-1)    # Shape (1,3) - 1 row, 3 cols

# 'a' stretches horizontally to (4,3), 'b' stretches vertically to (4,3)
# Resulting shape is (4,3)
print(f"Shape of vector + matrix broadcast (a + b): {(a + b).shape}")


# ==============================================================================
# TOPIC 2: DECISION TREES
# ==============================================================================

try:
    from public_tests import *
    from utils import *
except ImportError:
    print("\nNote: 'public_tests' and 'utils' not found. Visualization/testing functions will be skipped.")

print("\n==================================================")
print("TOPIC 2: DECISION TREES (Mushroom Classification)")
print("==================================================")

# ==============================================================================
# DATASET
# ==============================================================================
# Features:
# 0: Brown Color (1 = Brown cap, 0 = Red cap)
# 1: Tapering Shape (1 = Tapering Stalk Shape, 0 = Enlarging stalk shape)
# 2: Solitary (1 = Yes, 0 = No)
# Label:
# y = 1 (edible), y = 0 (poisonous)

X_train_dt = np.array([
    [1,1,1], [1,0,1], [1,0,0], [1,0,0], [1,1,1],
    [0,1,1], [0,0,0], [1,0,1], [0,1,0], [1,0,0]
])
y_train_dt = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

print("\n--- Data Setup ---")
print("First few elements of X_train_dt:\n", X_train_dt[:5])
print("First few elements of y_train_dt:", y_train_dt[:5])
print('The shape of X_train_dt is:', X_train_dt.shape)
print('The shape of y_train_dt is:', y_train_dt.shape)
print('Number of training examples (m):', len(X_train_dt))

# ==============================================================================
# PART 1: COMPUTE ENTROPY
# ==============================================================================
# Formula: H(p1) = -p1 * log2(p1) - (1 - p1) * log2(1 - p1)

def compute_entropy(y):
    """
    Computes the entropy for a node.
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (1) or poisonous (0)
       
    Returns:
        entropy (float): Entropy at that node
    """
    entropy = 0.
    
    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y) 
        if p1 != 0 and p1 != 1:
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
            entropy = 0. 
            
    return entropy

print("\nEntropy at root node: ", compute_entropy(y_train_dt))

# ==============================================================================
# PART 2: SPLIT DATASET
# ==============================================================================

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into left and right branches.
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices.
        feature (int):           Index of feature to split on.
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    left_indices = []
    right_indices = []
    
    for i in node_indices:   
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
            
    return left_indices, right_indices

print("\n--- Split Dataset Test ---")
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature = 0 # 0 (Brown Cap), 1 (Tapering Stalk), 2 (Solitary)

left_indices, right_indices = split_dataset(X_train_dt, root_indices, feature)
print(f"CASE 1 (Feature {feature}):")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# ==============================================================================
# PART 3: COMPUTE INFORMATION GAIN
# ==============================================================================
# Formula: Info Gain = H(node) - (w_left * H(left) + w_right * H(right))

def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information gain of splitting the node on a given feature.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices.
        feature (int):          Index of feature to split on
   
    Returns:
        information_gain (float): Computed info gain
    """    
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Extract subsets
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_entropy = w_left * left_entropy + w_right * right_entropy    
    
    information_gain = node_entropy - weighted_entropy
    
    return information_gain

print("\n--- Information Gain Tests ---")
print("Info Gain (Brown Cap):", compute_information_gain(X_train_dt, y_train_dt, root_indices, feature=0))
print("Info Gain (Tapering):", compute_information_gain(X_train_dt, y_train_dt, root_indices, feature=1))
print("Info Gain (Solitary):", compute_information_gain(X_train_dt, y_train_dt, root_indices, feature=2))

# ==============================================================================
# PART 4: GET BEST SPLIT
# ==============================================================================

def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature to split the node data.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         target variable
        node_indices (ndarray): List containing the active indices.
        
    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = 0

    for feature in range(num_features): 
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:  
            max_info_gain = info_gain
            best_feature = feature
            
    return best_feature

best_feature = get_best_split(X_train_dt, y_train_dt, root_indices)
print(f"\nBest feature to split on at root: {best_feature}")

# ==============================================================================
# PART 5: BUILDING THE TREE (RECURSIVE)
# ==============================================================================

tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using a recursive algorithm that splits the dataset into 2 subgroups at each node.
    """ 
    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " " * current_depth + "-" * current_depth
        print(f"{formatting} {branch_name} leaf node with indices {node_indices}")
        return
   
    # Get best split and split the data
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-" * current_depth
    print(f"{formatting} Depth {current_depth}, {branch_name}: Split on feature {best_feature}")
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # Continue splitting the left and the right child
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1)

print("\n--- Tree Construction Output ---")
build_tree_recursive(X_train_dt, y_train_dt, root_indices, "Root", max_depth=2, current_depth=0)
