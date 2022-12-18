import yaml
import tensorflow as tf

from helpers import *
from models import *


class Config:

    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, "r") as file_descriptor:
            data = yaml.load(file_descriptor, Loader=yaml.Loader)
        
        self.data = data
        
        # Loop parameters
        n = self.data["loop"]["n"]
        self.N = n
        
        verbose = self.data["loop"]["verbose"]
        self.VERBOSE = verbose
        
        # Model parameters
        model = self.data["model_params"]["model_name"]
        self.MODEL = eval(model)
        
        input_shape = self.data["model_params"]["input_shape"]
        self.INPUT_SHAPE = eval(input_shape)
        
        optimizer = self.data["model_params"]["optimizer"]
        self.OPTIMIZER = eval(optimizer)
        
        loss = self.data["model_params"]["loss"]
        self.LOSS = eval(loss)
        
        metrics = self.data["model_params"]["metrics"]
        self.METRICS = metrics
        
        # Data loading parameters
        name = self.data["data_loading_params"]["name"]
        self.NAME = name
        
        batch_size = self.data["data_loading_params"]["batch_size"]
        self.BATCH_SIZE = batch_size
        
        norm_func = self.data["data_loading_params"]["norm_func"]
        self.NORM_FUNC = eval(norm_func)
        
        resize = self.data["data_loading_params"]["resize"]
        if resize == 0:
            self.RESIZE = False
        else:
            self.RESIZE = eval(resize)
        
        custom_dir = self.data["data_loading_params"]["custom_dir"]
        if custom_dir == 0:
            self.CUSTOM_DIR = False
        else:
            self.CUSTOM_DIR = custom_dir
        
        validation_or_test = self.data["data_loading_params"]["validation_or_test"]
        self.VALIDATION_OR_TEST = validation_or_test
        
        pass
