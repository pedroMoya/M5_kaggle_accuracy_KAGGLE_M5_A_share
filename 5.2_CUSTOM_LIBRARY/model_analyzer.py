# Model architecture analyzer
import os
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import layers
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses, models
from tensorflow.keras import metrics
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as kb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model, model_to_dot

# open local settings
with open('./settings.json') as local_json_file:
    local_submodule_settings = json.loads(local_json_file.read())
    local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_submodule_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)


class model_structure:

    def analize(self, local_model_name, local_settings):
        try:
            # loading model (h5 format)
            print('trying to open model file (assuming h5 format)')
            local_model = models.load_model(''.join([local_settings['models_path'], local_model_name]))
            # saving architecture in JSON format
            local_model_json = local_model.to_json()
            with open(''.join([local_settings['models_path'], local_model_name,
                               '_analyzed_.json']), 'w') as json_file:
                json_file.write(local_model_json)
                json_file.close()
            # changing for subclassing to functional model
            local_model_json = json.loads(local_model_json)
            print(type(local_model_json))
            local_batch_size = None
            local_time_step_days = local_model_json['config']['build_input_shape'][1]
            local_features = local_model_json['config']['build_input_shape'][2]
            input_layer = layers.Input(batch_shape=(local_batch_size, local_time_step_days, local_features))
            prev_layer = input_layer
            for layer in local_model.layers:
                prev_layer = layer(prev_layer)
            functional_model = models.Model([input_layer], [prev_layer])
            # plotting (exporting to png) the model
            plot_path = ''.join([local_settings['models_path'], local_model_name, '_model.png'])
            # model_to_dot(functional_model, show_shapes=True, show_layer_names=True, rankdir='TB',
            #     expand_nested=True, dpi=96, subgraph=True)
            plot_model(functional_model, to_file=plot_path, show_shapes=True, show_layer_names=True,
                       rankdir='TB', expand_nested=True, dpi=216)
            plot_model(functional_model, to_file=''.join([plot_path, '.pdf']), show_shapes=True, show_layer_names=True,
                       rankdir='TB', expand_nested=True)
        except Exception as e1:
            print('Error reading or saving model')
            print(e1)
            return False
        return True
