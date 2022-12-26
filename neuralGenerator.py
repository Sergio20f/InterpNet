import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class createMultipleNeuralNetworks():
    
    def __init__(self, train_X, train_y, n_neural_networks=10,n_hidden_layers=0, n_categories=2, neuron_number_list = [512], activation_function_list=['relu', 'softmax'], loss_list=['binary_crossentropy'], optimizer_list=['adam'], epochs_list=[1], batch_size_list=[32], input_shape=(None, 500, 2)):
        self.train_X = train_X
        self.train_y = train_y
        self.n_neural_networks = int(n_neural_networks)
        self.n_hidden_layers = int(n_hidden_layers)
        self.n_categories = int(n_categories)
        self.neuron_number_list = neuron_number_list
        self.activation_function_list = list(activation_function_list)
        self.loss_list = list(loss_list)
        self.optimizer_list = list(optimizer_list)
        self.epochs_list = list(epochs_list)
        self.batch_size_list = list(batch_size_list)
        self.input_shape = input_shape
    
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
    
    def getConfigFiles(self, path = '\\Config\\ETL\\', config_name = ''):
        current_path = os.getcwd()
        folder_code = os.path.dirname(current_path)
        folder_working =  os.path.dirname(folder_code)
        config_path = folder_working + path + config_name
        return config_path
    
    def generate_model(self):
        model = Sequential()
        
        # Input layer
        model = self.add_dense_layer(model, input_layer=True)
        
        # Hidden layers
        for _ in range(self.n_hidden_layers):
            model = self.add_dense_layer(model,  input_layer=False)

        # Output layer
        model.add(Dense(self.n_categories, activation="softmax"))
        print('Random neural network architecture summary: ')
        model.summary()
        
        return model 
    
    def save_models(self, i, model):
        return model.save('neural_networks/model'+str(i)+".h5")
    
    def train(self, model):
        loss = np.random.choice(self.loss_list)
        optimizer = np.random.choice(self.optimizer_list)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        batch_size = np.random.choice(self.batch_size_list)
        epochs = np.random.choice(self.epochs_list)
        # Fitting model on training data
        model.fit(self.train_X, self.train_y, batch_size=batch_size, epochs=epochs, verbose=1)
        return model
    
    def main(self):
        for i in range(1, self.n_neural_networks):
            model = self.generate_model()
            model = self.train(model)
            self.save_models(i, model)
            
        neural_networks_list = [self.generate_model() for _ in range(self.n_neural_networks)]
        return neural_networks_list