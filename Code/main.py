import warnings
import autograd.numpy as np  
from sklearn.model_selection import train_test_split


from functions import (
    cost, create_layers_batch, feed_forward_batch, feed_forward_saver_batch,
    backpropagation_batch, polynomial_features, plot_accuracy
)
from activations import (
    sigmoid, ReLU, leaky_ReLu, sigmoid_der, ReLU_der, leaky_ReLU_der, softmax, softmax_der
)
from costs import (
    mse, mse_der, cross_entropy, binary_cross_entropy_loss,
    binary_cross_entropy_loss_l1, binary_cross_entropy_loss_l2,
    multiclass_cross_entropy_loss, crossentropy_der
)
from train_funct import (
    train_network, train_network_SGD_momentum, train_network_RMSprop, train_network_ADAM
)

def main():
    # Setting seed for reproducibility
    np.random.seed(42)
    n = 200

    # Creating a grid of n=200 points drawn from the random distribution
    x = np.random.uniform(-1, 1, n)

    # Setting up Runge's function, 1 normal and 1 with added noise
    y = 1.0 / (1 + 25 * x**2) + np.random.normal(0, 0.1, x.shape)

    # Calculating sample statistics
    x_mean = np.mean(x)
    x_variance = np.std(x, ddof=1) #sample variance

    #Scaling x values for a visual comparison
    x_st = (x - x_mean) / x_variance



    # Now we split the dataset for training and testing (using the standardized x values) and noisy Runge
    X_train, X_test, y_train, y_test = train_test_split(x_st, y, test_size=0.2, random_state = 42)

    # Define NN parameter
    poly_degree = 5
    inputs_train = np.array(polynomial_features(X_train, poly_degree))
    targets_train = y_train.reshape(-1, 1)

    inputs_test = np.array(polynomial_features(X_test, poly_degree))
    targets_test = y_test.reshape(-1, 1)
    #print('target', targets)
    n_iter = 1000
    lern_rate = 0.01
    network_input_size = poly_degree + 1
    layer_output_sizes = [100, 1]
    activation_funcs = [ sigmoid, ReLU]
    activation_ders = [ sigmoid_der, ReLU_der]

    #test that predict changes
    layers = create_layers_batch(network_input_size, layer_output_sizes)
    init_predictions = feed_forward_batch(inputs_train, layers, activation_funcs)
    print('initial prediction mse: ', mse(init_predictions, targets_train))
    
    # train network: can choose type of gradient there
    predictions, last_prediction, layers_appended = train_network_ADAM(inputs_train, layers, activation_funcs, targets_train, activation_ders, learning_rate=lern_rate, epochs=n_iter)
    print('train MSE: ',mse(targets_train,last_prediction))

    test_predict = feed_forward_batch(inputs_test, layers_appended, activation_funcs)
    print('test MSE: ',mse(test_predict,targets_test))


    mse_total = []
    for i in range(n_iter):
        mse_temp = mse(predictions[i], targets_train)
        mse_total.append([i,mse_temp])

    plot_accuracy(mse_total)

  


if __name__ == "__main__":
    main()
