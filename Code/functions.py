import autograd.numpy as np  # mirror original behavior
import matplotlib.pyplot as plt

from activations import (
    sigmoid, ReLU, leaky_ReLu,
    sigmoid_der, ReLU_der, leaky_ReLU_der,
    softmax, softmax_der
)
from costs import (
    mse, mse_der, cross_entropy, binary_cross_entropy_loss,
    binary_cross_entropy_loss_l1, binary_cross_entropy_loss_l2,
    multiclass_cross_entropy_loss, crossentropy_der
)

def cost(_input, layers, activation_funcs, target):
    predict = feed_forward_batch(_input, layers, activation_funcs)
    return mse(predict, target)

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    np.random.seed(42)
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        W = W.T
        b = np.random.randn(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers

def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

def feed_forward_saver_batch(inputs, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)
        zs.append(z)
    return layer_inputs, zs, a

def backpropagation_batch(inputs, layers, activation_funcs, target, activation_ders, cost_der):
    layer_inputs, zs, predict = feed_forward_saver_batch(inputs, layers, activation_funcs)
    layer_grads = [() for _ in layers]

    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        if i == len(layers) - 1:
            dC_da = cost_der(predict, target)
        else:
            (W, b) = layers[i + 1]
            dC_da = dC_dz @ W.T
        dC_dz = dC_da * activation_der(z)
        dC_dW = np.matmul(layer_input.T, dC_dz) / target.shape[0]
        dC_db = np.mean(dC_dz, axis=0)
        layer_grads[i] = (dC_dW, dC_db)
    return layer_grads

def polynomial_features(x, p):
    """Manually implementing PolynomialFeatures."""
    n = len(x)
    X = np.zeros((n, p + 1))
    X[:, 0] = 1
    for i in range(1, p + 1):
        X[:, i] = x ** i
    return X

def plot_accuracy(mse_total):
    n_iter = [point[0] for point in mse_total]
    mse_vals = [point[1] for point in mse_total]
    plt.figure(figsize=(12, 6))
    plt.plot(n_iter, mse_vals, marker='o')
    plt.title('Y Values by Iteration with Highlighted Drops')
    plt.xlabel('Number of Iterations')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def calculate_nlayer_nn(inputs_train, targets_train,inputs_test, targets_test, n_iter,learning_rates, network_input_size, layer_output_sizes, activation_funcs, activation_ders, train_network_type):
    results = []          # (lr, train_mse, test_mse)
    histories = {}        # lr -> [(epoch, mse), ...]
    for lr in learning_rates:
        # Fresh init for each LR
        layers = create_layers_batch(network_input_size, layer_output_sizes)
    
        # Initial prediction (before training)
        init_pred = feed_forward_batch(inputs_train, layers, activation_funcs)
        #print(f'[lr={lr}] initial train MSE: {mse(init_pred, targets_train):.6f}')
    
        # Train with SGD + momentum (your function)
        predictions, last_pred, trained_layers = train_network_type(
            inputs_train, layers, activation_funcs, targets_train, activation_ders,
            learning_rate=lr, epochs=n_iter
        )
    
        # Train/Test MSE after training
        train_mse = mse(last_pred, targets_train)
        test_pred = feed_forward_batch(inputs_test, trained_layers, activation_funcs)
        test_mse = mse(test_pred, targets_test)
        #print(f'[lr={lr}] final   train MSE: {train_mse:.6f} | test MSE: {test_mse:.6f}')
    
        # Store epoch-wise MSE history for plotting
        mse_total = []
        for i in range(n_iter):
            mse_total.append([i, mse(predictions[i], targets_train)])
        histories[lr] = mse_total
    
        # Collect summary row
        results.append((lr, float(train_mse), float(test_mse)))
    return results, histories

def print_output(results, gd_type):
    # ---- Pretty print a small summary ----
    print(f"\n=== Learning rate summary {gd_type} ===")
    for lr, tr, te in results:
        print(f"lr={lr:<8g}  train MSE={tr:.6f}  test MSE={te:.6f}")

def plot_MSE_to_Iter(title_text, histories):
    # ---- Plot training curves for all learning rates ----
    plt.figure()
    for lr, curve in histories.items():
        iters = [i for i, _ in curve]
        values = [v for _, v in curve]
        plt.plot(iters, values, label=f'lr={lr}')
    plt.xlabel('N iterations', fontsize = 25)
    plt.ylabel('MSE', fontsize = 25)
    plt.title(title_text, fontsize = 30)
    plt.legend()
    plt.tight_layout()
    plt.show()