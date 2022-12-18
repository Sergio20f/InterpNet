import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import scipy
from scipy import optimize
import pandas as pd
from datetime import datetime


from helpers import normalize_img, monoExp, powerlaw, plot_fits
from data_loading import Data


def build_and_compile(input_shape, optimizer, loss, metrics, model, lr=0.001):
    """
    Function that builds the model instance and compiles it.
    In future work we will implement further customisation for the model architecture.

    :param model: Tensorflow model. If "None" then the function will construct a default simple CNN with input_shape.
    :param input_shape: Input shape as a tuple for the first convolutional layer of the model. Default: (28, 28, 1).
    :type input_shape: tuple
    :param optimizer: Optimizer to compile the model.
    :param lr: Learning rate.
    :type lr: float
    :param loss: Loss used to compile the model.
    :param metrics: Metrics to compute the performance of the model throughout training.

    :return: Compiled model for further use in training.
    """

    # Compile the model
    model.compile(optimizer=optimizer(lr),# momentum=0.9), # Remove momentum if not SGD
                  loss=loss,
                  metrics=metrics)

    return model


def training_fit_loop(model, epochs, model_params: tuple, n: int, data_loading_params: tuple, N=1, plot=True, verbose=0):
    """
    Function that will put together the previously defined functions with the aim of generating a training loops with n
    iterations. It will also use the results of the training experiments and fit it to both an exponential and a power
    law function determining which one is a better fit. Lastly, it will generate a dataframe with the results of each
    experiment with some interesting metrics to look at.

    :param model: Tensorflow model to train. If None, then the function uses a default simple CNN.
    :param model_params: Parameters to feed build_and_compile(). Tuple elements must be in specific order -> 1. input_shape, 2.
    optimizer, 3. loss, 4. metrics.
    :param data_step: How many data points are gonna be added to the dataset after each training step.
    :type data_step: int
    :param n: Number of iterations for training in each experiment.
    :param n: int
    :param data_loading_params: Data loading parameters in specific order -> 1. name, 2. batch_size, 3. norm_func, 4. resize, 5. 
    custom_dir, 6. val_or test
    :type data_loading_params: tuple
    :param N: Number of experiments.
    :type N: int
    :param start_data: Determines the size of the initial training dataset from 0 (Default: 500).
    :type start_data: int
    :param plot: Boolean parameter. If true then the function will plot fitting results. If false, it will just return
    the results.
    :type plot: bool
    :param save_df: If not False, path to folder where the dataframe will be saved as a pickle.
    :type save_df: False or str

    :return: R² of the fit, list of the fitting parameters, dataframe with useful information from the fitting process.
    """
    # Decompose data loading parameters
    name, batch_size, norm_func, resize, custom_dir, validation_or_test = data_loading_params
    input_shape, optimizer, loss, metrics = model_params

    data = Data(name=name, batch_size=batch_size, norm_func=norm_func, resize=resize, custom_dir=custom_dir)

    if custom_dir:
        # Will only run if we are using a custom dataset
        test_data = data.test_data_prep(validation_or_test=validation_or_test)
        #test_data = test_data.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE) # Temporary
        validation_data = test_data # Test and validation data will be the same unless changed
        
    else:
        validation_data, test_data = data.test_data_prep(validation_or_test=validation_or_test)

    # Loop repeats the process for N experiments.
    for j in range(N):
        # Here we could run different training loops for different models - For a single model, N = 1, two different models N = 2, etc.
        for i in range(n):
            # Define the model so it starts training from scratch
            model = build_and_compile(model=model, input_shape=input_shape, optimizer=optimizer, loss=loss, metrics=metrics) # Will need to change the parameters
            weights = model.get_weights()
            
            train_data = data.train_data_prep()

            history = model.fit(train_data, epochs=epochs, validation_data=validation_data, verbose=verbose)

            # Save model weights as .h5 file
            model.save_weights(f"model_{i}_weights.h5")
            # Save model weights as a .npy file
            np.save(f"model_{i}_weights.npy", model.get_weights())

            model.set_weights(weights)

