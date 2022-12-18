import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Accuracy
import numpy as np

class createMultipleNeuralNetworks():
    
    def __init__(self, n_neural_networks=10, input_shape=(500, 2), n_hidden_layers=0, n_categories=2, neuron_number_list = [2**4, 2**6, 2**8], activation_function_list=['relu', 'softmax', 'tanh']):
        self.n_neural_networks = n_neural_networks
        self.input_shape = input_shape
        self.n_hidden_layers = n_hidden_layers
        self.n_categories = n_categories
        self.neuron_number_list = neuron_number_list
        self.activation_function_list = activation_function_list
    
    def add_dense_layer(self, model, input_layer=False):
        # Random number of layers, and activation function
        neuron_number = np.random.choice(self.neuron_number_list)
        print("Number of nuerons: ", neuron_number)
        activation_function = np.random.choice(self.activation_function_list)
        print("Activation Function: ", activation_function)
        
        if input_layer is True:
            model.add(Dense(neuron_number, activation=activation_function, input_shape= self.input_shape))   
        else:
            model.add(Dense(neuron_number, activation=activation_function))   
        
        return model
    
    def generate_model(self):
        model = Sequential()
        
        # Input layer
        model = self.add_dense_layer(model, input_layer=True)
        
        for _ in range(self.n_hidden_layers):
            model = self.add_dense_layer(model)

        # Output layer
        model.add(Dense(self.n_categories, activation="softmax"))
        print('Random neural network architecture summary: ')
        model.summary()
        
        return model 
    
    def main(self):
        neural_networks_list = [self.generate_model() for _ in range(self.n_neural_networks)]
        return neural_networks_list
        
        