import json
import os
class utils:
    def __init__(self):
        pass

    def get_inputs_from_json(self, path = '\\configs\\', config_name = 'configurations.json'):
        current_path = os.getcwd()
        config_path = current_path + path + config_name 
        with open(config_path) as f:
                file = json.load(f)
                return list(file['Neural_Network_Generator'].values())