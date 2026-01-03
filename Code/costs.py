import autograd.numpy as np  

def mse(predict, target, W= 0, lambda_l=0):
    """MSE """
    return np.mean((predict - target) ** 2)

def mse_l1(y_true, y_pred, weights, lambda_l1=0.01):
    """MSE + L1 regularization."""
    return mse(y_true, y_pred) + lambda_l1 * np.sum(np.abs(weights))

def mse_l2(y_true, y_pred, weights, lambda_l2=0.01):
    """MSE + L2 regularization."""
    return mse(y_true, y_pred) + lambda_l2 * np.sum(weights ** 2)


def mse_der(predict, target, W= 0, lambda_l=0):
    n = target.shape[0]
    return (2.0 / n) * (predict - target)

def mse_l1_der(y_pred, y_true, W, lambda_l1=0.01):
    """
    Derivatives for MSE + L1 regularization.
    Returns:
      dL_dy_pred : gradient wrt predictions
      dL_dW      : additional regularization gradient wrt weights
    """
    dL_dy_pred = mse_der(y_true, y_pred)
    dL_dW = lambda_l1 * np.sign(W)
    return dL_dy_pred, dL_dW


def mse_l2_der(y_pred, y_true, W, lambda_l2=0.01):
    """
    Derivatives for MSE + L2 regularization.
    Returns:
      dL_dy_pred : gradient wrt predictions
      dL_dW      : additional regularization gradient wrt weights
    """
    dL_dy_pred = mse_der(y_true, y_pred)
    dL_dW = 2.0 * lambda_l2 * W
    return dL_dy_pred, dL_dW


def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def binary_cross_entropy_loss(predict, target):
    return np.mean(target * np.log(predict) + (1 - target) * np.log(1 - predict))

def binary_cross_entropy_loss_l1(predict, target, weights, lambda_l1=0.01):
    loss = -binary_cross_entropy_loss(predict, target)
    regularization_loss = lambda_l1 * np.sum(np.abs(weights))
    return loss + regularization_loss

def binary_cross_entropy_loss_l2(predict, target, weights, lambda_l2=0.01):
    loss = -binary_cross_entropy_loss(predict, target)
    regularization_loss = (lambda_l2 / 2) * np.sum(weights ** 2)
    return loss + regularization_loss

def multiclass_cross_entropy_loss(predict, target):
    return np.sum(target * np.log(predict)) / target

def crossentropy_der(predict, target):
    return predict - target


# Custom derivative for binary cross-entropy w.r.t. activations (predictions)
def binary_cross_entropy_der(predict, target):
    """Derivative of BCE loss w.r.t. activations (a = predict).

    L = -1/n * sum[y*log(a) + (1-y)*log(1-a)]
    dL/da = (a - y) / (a*(1-a) + eps) / n
    """
    eps = 1e-8
    n = target.shape[0]
    return (predict - target) / (predict * (1.0 - predict) + eps) / n
