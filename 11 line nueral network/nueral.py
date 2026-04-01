import numpy as np


def nonlin(x, deriv=False):
    """
    Sigmoid activation function.
    
    Args:
        x: Input array
        deriv: If True, returns derivative of sigmoid
    
    Returns:
        Sigmoid activation or its derivative
    """
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# ===== Data Preparation =====
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ===== Network Initialization =====
np.random.seed(1)

# Hidden layer weights (3 inputs -> 4 hidden units)
weights_hidden = 2 * np.random.random((3, 4)) - 1

# Output layer weights (4 hidden units -> 1 output)
# Adding another layer to the neural network brings it to the realm of deep learning
weights_output = 2 * np.random.random((4, 1)) - 1

# ===== Training Parameters =====
ITERATIONS = 60000
OUTPUT_FREQUENCY = 10000

# ===== Training Loop =====
for iteration in range(ITERATIONS):
    # Forward pass
    layer0_output = X
    layer1_output = nonlin(np.dot(layer0_output, weights_hidden))
    layer2_output = nonlin(np.dot(layer1_output, weights_output))

    # Calculate error
    layer2_error = y - layer2_output

    # Print progress
    if iteration % OUTPUT_FREQUENCY == 0:
        print(f"Iteration {iteration}: error: {np.mean(np.abs(layer2_error)):.6f}")

    # Backward pass - calculate deltas
    layer2_delta = layer2_error * nonlin(layer2_output, deriv=True)

    layer1_error = layer2_delta.dot(weights_output.T)
    layer1_delta = layer1_error * nonlin(layer1_output, deriv=True)

    # Update weights
    weights_output += layer1_output.T.dot(layer2_delta)
    weights_hidden += layer0_output.T.dot(layer1_delta)

print("\n=== Training Complete ===")
print(f"Final output predictions:\n{layer2_output}")