"""
Personal Lab Notes: Neural Networks (Binary Classification)
Recognizing Handwritten Digits (0 and 1)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Reduce TensorFlow verbosity for cleaner output
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# ==============================================================================
# DATA SETUP (Mock Data for Standalone Testing)
# ==============================================================================
# X shape: (1000, 400) - 1000 examples of 20x20 pixel images unrolled into 400 features.
# y shape: (1000, 1) - Labels: 0 (digit zero) or 1 (digit one).

print("\n--- Data Setup ---")
# Generating mock data to represent 4 examples of 400 pixels
X_train = np.random.rand(4, 400) 
y_train = np.array([[0], [1], [0], [1]]) 

print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is:', y_train.shape)


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
    ], name = "my_model" 
)

# View model architecture and parameter counts
model.summary()

# 2. Compile the model (Define Loss and Optimizer)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

# 3. Train the model
model.fit(X_train, y_train, epochs=5, verbose=0) # verbose=0 hides training output

# 4. Make Predictions
# Predict on the first example (needs to be reshaped to 2D array: 1 example, 400 features)
prediction = model.predict(X_train[0].reshape(1,400))
print(f"Raw prediction probability: {prediction[0][0]:.4f}")

# Apply threshold
yhat = 1 if prediction >= 0.5 else 0
print(f"Prediction after threshold: {yhat}")


# ==============================================================================
# PART 2: NUMPY IMPLEMENTATION (Under the Hood - Unvectorized)
# ==============================================================================
# Building a Dense layer from scratch to understand the math.
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
# PART 3: VECTORIZED NUMPY IMPLEMENTATION (Faster, Matrix Math)
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
Predictions = my_sequential_v(X_train, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)
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