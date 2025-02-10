import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        
        # Weights between input and hidden layer
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Weights between hidden and output layer
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Derivative of sigmoid function

    def forward(self, X):
        """ Forward propagation """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # Output

        return self.a2

    def backward(self, X, y):
        """ Backpropagation """
        m = X.shape[0]  # Number of samples

        # Error in output layer
        error_output = self.a2 - y
        delta_output = error_output * self.sigmoid_derivative(self.a2)

        # Error in hidden layer
        error_hidden = delta_output.dot(self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Gradient descent weight updates
        self.W2 -= self.learning_rate * np.dot(self.a1.T, delta_output) / m
        self.b2 -= self.learning_rate * np.sum(delta_output, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * np.dot(X.T, delta_hidden) / m
        self.b1 -= self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True) / m

    def train(self, X, y, epochs=10000):
        """ Training loop """
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.a2))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """ Predict function """
        return self.forward(X)

# XOR Dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Inputs
y = np.array([[0], [1], [1], [0]])  # Expected outputs

# Initialize and train the neural network
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)
nn.train(X, y, epochs=10000)

# Predictions
predictions = nn.predict(X)
print("\nPredictions:")
print(predictions)