import autograd.numpy as np
from activations import sigmoid  # only what's used

def add_intercept(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def fit(X, y, learning_rate=0.5, num_iterations=2000):
    n_data, n_features = X.shape
    beta = np.zeros(n_features)
    y = y.astype(float)
    for _ in range(num_iterations):
        linear = X @ beta
        y_hat = sigmoid(linear)
        gradient = (X.T @ (y_hat - y)) / n_data
        beta -= learning_rate * gradient
    return beta

def predict(X, beta, threshold=0.5):
    probs = sigmoid(X @ beta)
    return (probs >= threshold).astype(int)

# and gates
X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1], dtype=float)

X = add_intercept(X)
beta = fit(X, y, learning_rate=0.5, num_iterations=2000)
preds = predict(X, beta)

print("Predictions:", preds.tolist())
print("Target:", y)
