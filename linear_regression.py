"""
Personal Lab Notes: Linear Regression & Logistic Regression
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# ==============================================================================
# PART 1: LINEAR REGRESSION
# ==============================================================================

# Data Loading
x_train = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598])
y_train = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233])

# Check the types and view the first few elements
print("Type of x_train:", type(x_train))
print("First five elements of x_train are:\n", x_train[:5])
print("First five elements of y_train are:\n", y_train[:5])  

# Check the dimensions (shape) of the arrays to ensure they match
print('The shape of x_train is:', x_train.shape)
print('The shape of y_train is: ', y_train.shape)
print('Number of training examples (m):', len(x_train))

# Visualize the data (scatter plot)
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show() # Displays the graph


# Core Machine Learning Functions (Linear Regression):
# 1. Cost Function: Measures how wrong the model's predictions are
# 2. Gradient Function: Calculates the direction to adjust the parameters to reduce error

def compute_cost(x, y, w, b): 
    """
    Computes the Mean Squared Error (MSE) cost function for linear regression.
    Cost represents the average squared difference between predictions and actual values.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population) 
        y (ndarray): Shape (m,) Label (Actual profits)
        w, b (scalar): Parameters of the model (Weight and Bias)
    
    Returns:
        total_cost (float): The cost of using w,b as the parameters
    """
    m = x.shape[0] # Number of training examples
    cost_sum = 0
    
    # Loop through every data point to calculate the error
    for i in range(m):
       f_wb = w * x[i] + b           # 1. Make a prediction (y = wx + b)
       cost = (f_wb - y[i]) ** 2     # 2. Calculate the squared error for this point
       cost_sum = cost_sum + cost    # 3. Add it to the running total

    # Divide by 2m to get the final averaged cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# Test the cost function with initial guesses for w and b
initial_w, initial_b = 2, 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(f'Cost at initial w=2, b=1: {cost:.3f}')


def compute_gradient(x, y, w, b): 
    """
    Computes the partial derivatives (gradients) of the cost function.
    These derivatives tell us how to change 'w' and 'b' to decrease the cost.
    
    Args:
      x (ndarray): Shape (m,) Input to the model 
      y (ndarray): Shape (m,) Label 
      w, b (scalar): Parameters of the model  
      
    Returns:
      dj_dw (scalar): Gradient of the cost with respect to parameter w
      dj_db (scalar): Gradient of the cost with respect to parameter b     
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
       f_wb = w * x[i] + b                 # 1. Make prediction
       dj_dw_i = (f_wb - y[i]) * x[i]      # 2. Derivative w.r.t 'w'
       dj_db_i = f_wb - y[i]               # 3. Derivative w.r.t 'b'

       dj_db += dj_db_i
       dj_dw += dj_dw_i

    # Average the gradients over all examples
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_dw, dj_db

# Test the gradient function
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, 0, 0)
print(f'Gradient at initial w=0, b=0: dw={tmp_dj_dw:.3f}, db={tmp_dj_db:.3f}')


# Optimization (Gradient Descent):
# Defines the loop that iteratively updates the parameters (w and b)
# using the gradients calculated above to find the optimal line of best fit

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn optimal parameters (w, b).
    
    Args:
      x, y (ndarray): Data inputs and labels
      w_in, b_in (scalar): Initial guesses for parameters
      cost_function, gradient_function: The functions defined in Section 4
      alpha (float): Learning rate (how big of a step to take)
      num_iters (int): Number of times to run the update loop
      
    Returns:
      w, b: Updated, optimized parameters
      J_history, w_history: Lists keeping track of cost and w over time for plotting
    """
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in) 
    b = b_in
    
    for i in range(num_iters):
        # 1. Calculate the gradients for the current w and b
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # 2. Update Parameters by taking a small step (alpha) in the opposite direction of the gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # 3. Track the cost to make sure it is decreasing
        if i < 100000:      
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        # 4. Print progress every 10% of the total iterations
        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
        
    return w, b, J_history, w_history


# Training the model: the alogrithm on the data to find the best w and b

# Initial guesses
initial_w = 0.
initial_b = 0.

# Hyperparameters
iterations = 1500
alpha = 0.01  # A small learning rate

# Run Gradient Descent
print("\n--- Starting Gradient Descent ---")
w_final, b_final, J_hist, w_hist = gradient_descent(
    x_train, y_train, initial_w, initial_b, 
    compute_cost, compute_gradient, alpha, iterations
)
print(f"Final parameters found: w={w_final:.3f}, b={b_final:.3f}\n")


# Evaluation // Visualization:
# With the best w and b, calculate the model's predicitons for
# every data point and plot the resulting line of best fit

m = x_train.shape[0]
predicted = np.zeros(m)

# Generate predictions for all data using the optimized w and b
for i in range(m):
    predicted[i] = w_final * x_train[i] + b_final

# Plot the original data as red X's
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Data') 

# Plot the model's predictions as a blue line
plt.plot(x_train, predicted, c="b", label='Linear Fit')

plt.title("Profits vs. Population per city (Model Fit)")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.legend()
plt.show()


# Making predicitons on new data:
# Use the trained model to predict outcomes for new data

# Example 1: Predict profit for a city with 35,000 people (Input x = 3.5)
predict1 = 3.5 * w_final + b_final
print(f"For population = 35,000, we predict a profit of ${predict1*10000:.2f}")

# Example 2: Predict profit for a city with 70,000 people (Input x = 7.0)
predict2 = 7.0 * w_final + b_final
print(f"For population = 70,000, we predict a profit of ${predict2*10000:.2f}")


# ==============================================================================
# PART 2: LOGISTIC REGRESSION (Classification)
# ==============================================================================
# Note: Redefining compute_cost, compute_gradient, and gradient_descent below
# updates them to use Logistic Regression formulas (Log-Loss instead of MSE).

def sigmoid(z):
    """
    Compute the sigmoid of z
    
    Args:
        z (scalar or ndarray): A scalar or numpy array of any size.
    Returns:
        g (scalar or ndarray): sigmoid(z), with the same shape as z.
    """
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, *argv):
    """
    Computes the Log-Loss cost over all examples for logistic regression.
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
    Returns:
      total_cost : (scalar) cost 
    """
    m, n = X.shape
    loss_sum = 0 

    for i in range(m): 
        z_wb = 0 
        for j in range(n): 
            z_wb += w[j] * X[i][j] 
        z_wb += b 

        f_wb = sigmoid(z_wb)
        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        loss_sum += loss

    total_cost = (1 / m) * loss_sum  
    return total_cost

def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression.
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = 0
        for j in range(n): 
            z_wb += X[i, j] * w[j]
        z_wb += b
        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i

        for j in range(n):
            dj_dw_ij = (f_wb - y[i]) * X[i, j]
            dj_dw[j] += dj_dw_ij

    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_db, dj_dw

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b.
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    m, n = X.shape   
    p = np.zeros(m)
   
    for i in range(m):   
        z_wb = 0
        for j in range(n): 
            z_wb += X[i, j] * w[j]
        z_wb += b
        
        f_wb = sigmoid(z_wb)
        p[i] = f_wb >= 0.5
        
    return p


# ==============================================================================
# PART 3: REGULARIZED LOGISTIC REGRESSION
# ==============================================================================

def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples for regularized logistic regression.
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost 
    """
    m, n = X.shape
    
    # Calls the compute_cost function implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    reg_cost = 0.
    for j in range(n):
        reg_cost_j = w[j]**2 
        reg_cost += reg_cost_j
        
    # Multiply by lambda / 2m outside the loop
    reg_cost = (lambda_ / (2 * m)) * reg_cost
    
    total_cost = cost_without_reg + reg_cost
    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for logistic regression with regularization.
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    """
    m, n = X.shape
    
    # Calculate the gradient without regularization first
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    
    # Add the regularization term to dj_dw
    for j in range(n): 
        dj_dw_j_reg = (lambda_ / m) * w[j] 
        dj_dw[j] = dj_dw[j] + dj_dw_j_reg
        
    return dj_db, dj_dw

# Optimization algorithm extended to support lambda_ for regularization
def gradient_descent_reg(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha. Includes lambda_ for reg.
    """
    m = len(x)
    J_history = []
    w_history = []
    
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(x, y, w, b, lambda_)   

        # Update Parameters
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db              
       
        # Save cost J at each iteration
        if i < 100000:      
            cost = cost_function(x, y, w, b, lambda_)
            J_history.append(cost)

        # Print cost at intervals
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
        
    return w, b, J_history, w_history