import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def target_function(x, y):
    return np.sin(x) * np.cos(y)

# Generate input data
x_values = np.linspace(-np.pi, np.pi, 100)
y_values = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x_values, y_values)
Z = target_function(X, Y)

# Flatten the data for training
inputs = np.vstack([X.ravel(), Y.ravel()]).T
outputs = Z.ravel()

# Number of input variables
n = inputs.shape[1]
# Number of neurons in the hidden layer (2n + 1)
hidden_units = 2 * n + 1

# Input layer
input_layer = tf.keras.layers.Input(shape=(n,))

# Hidden layer applying ψ functions
psi_outputs = []
for _ in range(hidden_units):
    psi = tf.keras.layers.Dense(n, activation='tanh')(input_layer)
    psi_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(psi)
    psi_outputs.append(psi_sum)

# Concatenate all ψ sums
concat_layer = tf.keras.layers.Concatenate()(psi_outputs)

# Output layer applying φ functions
output_layer = tf.keras.layers.Dense(1, activation='tanh')(concat_layer)

# Build the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(inputs, outputs, epochs=100, batch_size=16, verbose=0)


# Predict outputs
predicted_outputs = model.predict(inputs)

# Reshape for visualization
Z_pred = predicted_outputs.reshape(X.shape)

# Plot the target function and the approximation
fig = plt.figure(figsize=(14, 6))

# Original function
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Original Function')

# KANN approximation
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z_pred, cmap='viridis')
ax2.set_title('KANN Approximation')

plt.show()
