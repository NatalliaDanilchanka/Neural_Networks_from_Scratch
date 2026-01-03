import autograd.numpy as np  

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    return np.where(z > 0, z, 0)

def leaky_ReLu(z):
    alpha = 0.01
    return np.where(z > 0, z, alpha * z)

def sigmoid_der(z):
    sigm = sigmoid(z)
    return sigm * (1 - sigm)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU_der(z):
    alpha = 0.01
    return np.where(z >= 0, 1, alpha)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

def softmax_der(z):
    return 1
