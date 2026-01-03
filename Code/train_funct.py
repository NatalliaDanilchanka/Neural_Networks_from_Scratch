import autograd.numpy as np  # mirror original behavior

from costs import mse, mse_der
from functions import feed_forward_batch, backpropagation_batch

def train_network(_inputs, _layers, activation_funcs, targets, activation_ders,
                  learning_rate=None, momentum=None, epochs=None, cost_der=mse_der):
    """
    Plane gradient descend 
    """
    predictions = []
    prediction = _inputs
    tolerance = 1e-8
    for i in range(epochs):
        layers_updated = []
        layers_grad = backpropagation_batch(
            _inputs, _layers, activation_funcs, targets, activation_ders, cost_der
        )
        #print('MSE: \n', mse(feed_forward_batch(_inputs, _layers, activation_funcs), targets))
        for (W, b), (W_g, b_g) in zip(_layers, layers_grad):
            if tolerance is not None and np.linalg.norm(W_g) < tolerance and np.linalg.norm(b_g) < tolerance:
                print(f"Converged at iteration {i + 1}")
                break
            W -= W_g * learning_rate
            b -= b_g * learning_rate
            layers_updated.append((W, b))
        prediction = feed_forward_batch(_inputs, layers_updated, activation_funcs)
        predictions.append(prediction)
        _layers = layers_updated
    last_prediction = predictions[-1]

    return predictions, last_prediction, layers_updated


def train_network_SGD_momentum(_inputs, _layers, activation_funcs, targets, activation_ders,
                               learning_rate=0.001, momentum=0.8, epochs=30, cost_der=mse_der):
    """
    SGD with classical momentum. Includes per-parameter tolerance check.
    """
    tolerance = 1e-8
    predictions = []
    prediction = _inputs
    velocities = [(np.zeros(W.shape), np.zeros(b.shape)) for W, b in _layers]
    for i in range(epochs):
        layers_updated = []
        layers_grad = backpropagation_batch(
            _inputs, _layers, activation_funcs, targets, activation_ders, cost_der
        )
        for j, ((W, b), (W_g, b_g)) in enumerate(zip(_layers, layers_grad)):
            if tolerance is not None and np.linalg.norm(W_g) < tolerance and np.linalg.norm(b_g) < tolerance:
                print(f"Converged at iteration {i + 1}")
                break
            v_W, v_b = velocities[j]
            v_W = momentum * v_W - learning_rate * W_g
            v_b = momentum * v_b - learning_rate * b_g
            W += v_W
            b += v_b
            velocities[j] = (v_W, v_b)
            layers_updated.append((W, b))
        prediction = feed_forward_batch(_inputs, layers_updated, activation_funcs)
        predictions.append(prediction)
        _layers = layers_updated
    last_prediction = predictions[-1]
    return predictions, last_prediction, layers_updated


def train_network_RMSprop(_inputs, _layers, activation_funcs, targets, activation_ders,
                          learning_rate=0.001, rho=0.9, eps=1e-8, epochs=30,
                          grad_tol=1e-6, verbose=False, cost_der=mse_der):
    """
    RMSprop with early convergence when ||grad||_2 <= grad_tol.
    Pads predictions to 'epochs' by repeating the last prediction if stopped early.
    """
    predictions = []
    sq_grads = [(np.zeros_like(W), np.zeros_like(b)) for W, b in _layers]
    stopped_at = None

    for e in range(epochs):
        layers_grad = backpropagation_batch(
            _inputs, _layers, activation_funcs, targets, activation_ders, cost_der
        )

        # Early stop on global gradient norm
        grad_sq_sum = 0.0
        for (gW, gb) in layers_grad:
            grad_sq_sum += np.sum(gW * gW) + np.sum(gb * gb)
        grad_norm = np.sqrt(grad_sq_sum)
        if grad_norm <= grad_tol:
            if verbose:
                print(f"[RMSprop] Early stop at epoch {e+1}: ||grad||={grad_norm:.3e} ≤ {grad_tol:.3e}")
            stopped_at = e + 1
            prediction = feed_forward_batch(_inputs, _layers, activation_funcs)
            predictions.append(prediction)
            break

        layers_updated = []
        for j, ((W, b), (W_g, b_g)) in enumerate(zip(_layers, layers_grad)):
            W_sqw, b_sqw = sq_grads[j]
            W_sqw = rho * W_sqw + (1.0 - rho) * (W_g * W_g)
            b_sqw = rho * b_sqw + (1.0 - rho) * (b_g * b_g)

            W = W - learning_rate * W_g / (np.sqrt(W_sqw) + eps)
            b = b - learning_rate * b_g / (np.sqrt(b_sqw) + eps)

            sq_grads[j] = (W_sqw, b_sqw)
            layers_updated.append((W, b))

        _layers = layers_updated
        prediction = feed_forward_batch(_inputs, _layers, activation_funcs)
        predictions.append(prediction)

    if stopped_at is not None and stopped_at < epochs:
        last_pred = predictions[-1]
        predictions.extend([last_pred] * (epochs - stopped_at))
   
    last_prediction = predictions[-1]
    return predictions, last_prediction, layers_updated


def train_network_ADAM(_inputs, _layers, activation_funcs, targets, activation_ders,
                       learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, epochs=30,
                       grad_tol=1e-6, verbose=False, cost_der=mse_der):
    """
    Adam with early convergence when ||grad||_2 <= grad_tol.
    Pads predictions to 'epochs' by repeating the last prediction if stopped early.
    """
    predictions = []
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in _layers]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in _layers]

    t = 0
    stopped_at = None

    for e in range(epochs):
        t += 1
        layers_grad = backpropagation_batch(
            _inputs, _layers, activation_funcs, targets, activation_ders, cost_der
        )

        # Early stop on global gradient norm
        grad_sq_sum = 0.0
        for (gW, gb) in layers_grad:
            grad_sq_sum += np.sum(gW * gW) + np.sum(gb * gb)
        grad_norm = np.sqrt(grad_sq_sum)
        if grad_norm <= grad_tol:
            if verbose:
                print(f"[Adam] Early stop at epoch {e+1}: ||grad||={grad_norm:.3e} ≤ {grad_tol:.3e}")
            stopped_at = e + 1
            prediction = feed_forward_batch(_inputs, _layers, activation_funcs)
            predictions.append(prediction)
            break

        layers_updated = []
        for j, ((W, b), (gW, gb)) in enumerate(zip(_layers, layers_grad)):
            mW, mb = m[j]
            vW, vb = v[j]

            mW = beta1 * mW + (1.0 - beta1) * gW
            mb = beta1 * mb + (1.0 - beta1) * gb

            vW = beta2 * vW + (1.0 - beta2) * (gW * gW)
            vb = beta2 * vb + (1.0 - beta2) * (gb * gb)

            mW_hat = mW / (1.0 - beta1 ** t)
            mb_hat = mb / (1.0 - beta1 ** t)
            vW_hat = vW / (1.0 - beta2 ** t)
            vb_hat = vb / (1.0 - beta2 ** t)

            W = W - learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
            b = b - learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

            m[j] = (mW, mb)
            v[j] = (vW, vb)
            layers_updated.append((W, b))

        _layers = layers_updated
        prediction = feed_forward_batch(_inputs, _layers, activation_funcs)
        predictions.append(prediction)

    if stopped_at is not None and stopped_at < epochs:
        last_pred = predictions[-1]
        predictions.extend([last_pred] * (epochs - stopped_at))

    last_prediction = predictions[-1]
    return predictions, last_prediction, layers_updated
