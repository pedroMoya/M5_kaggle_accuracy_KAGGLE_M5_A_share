# functionality that builds local_bxy_x_train and local_bxy_y_train
# train and forecast time_series one by one_approach here is accumulated frequencies
import os
import sys
import logging
import logging.handlers as handlers
import json
import itertools as it
import pandas as pd
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
from tensorflow.keras.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

# open local settings
with open('./settings.json') as local_bxy_json_file:
    local_bxy_submodule_settings = json.loads(local_bxy_json_file.read())
    local_bxy_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_bxy_submodule_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# set random seed for reproducibility --> done in _2_train.py module
np.random.seed(42)
tf.random.set_seed(42)

# clear session
kb.clear_session()

# functions definitions


# classes definitions
class local_bxy_x_y_builder:

    def build_x_y_train_arrays(self, local_bxy_unit_sales, local_bxy_settings_arg, local_bxy_hyperparameters):
        try:
            # creating x_train and y_train arrays
            local_bxy_nof_series = local_bxy_unit_sales.shape[0]
            local_bxy_nof_selling_days = local_bxy_unit_sales.shape[1]
            local_bxy_last_learning_day_in_year = np.mod(local_bxy_nof_selling_days, 365)
            local_bxy_max_selling_time = local_bxy_settings_arg['max_selling_time']
            local_bxy_days_in_focus_frame = local_bxy_hyperparameters['days_in_focus_frame']
            local_bxy_window_input_length = local_bxy_hyperparameters['moving_window_input_length']
            local_bxy_window_output_length = local_bxy_hyperparameters['moving_window_output_length']
            local_bxy_moving_window_length = local_bxy_window_input_length + local_bxy_window_output_length
            local_bxy_time_steps_days = local_bxy_hyperparameters['time_steps_days']
            print('time_steps_days', local_bxy_time_steps_days)
            # nof_moving_windows = np.int32(nof_selling_days / moving_window_length)  # not used but useful (:-/)

            # checking consistence within time_step_days and moving_window
            if local_bxy_time_steps_days != local_bxy_moving_window_length:
                print('time_steps_days and moving_window_length are not equals, '
                      'that is not consistent with the preprocessing')
                print('please check it: reconcile values or rewrite the code in x_train '
                      'and y_train generation for this change')
                return False  # a controlled error will occur
            local_bxy_nof_years = local_bxy_settings_arg['number_of_years_ceil']
            local_bxy_stride_window_walk = local_bxy_hyperparameters['stride_window_walk']

            # ensure that each time_step has the same length (shape of source data) with no data loss at border condition..
            # simply concatenate at the start of the array the mean of the last 2 time_steps, to complete the shape needed
            print('dealing with last time_step_days')
            rest = np.ceil(local_bxy_nof_selling_days /
                           local_bxy_time_steps_days) * local_bxy_time_steps_days - local_bxy_nof_selling_days
            if rest != 0:
                local_bxy_mean_block = np.zeros(shape=(local_bxy_nof_series, int(rest)))
                for time_serie in range(local_bxy_nof_series):
                    local_bxy_mean_block[time_serie, :] = np.mean(local_bxy_unit_sales[time_serie,
                                                                  - 2 * local_bxy_time_steps_days:])
                local_bxy_unit_sales = np.concatenate((local_bxy_mean_block, local_bxy_unit_sales), axis=1)
                print(local_bxy_unit_sales.shape)

            # checking that adjustments was done well
            local_bxy_nof_selling_days = local_bxy_unit_sales.shape[1]  # necessary as local_bxy_unit_sales was concatenated
            local_bxy_remainder_days = np.mod(local_bxy_nof_selling_days, local_bxy_moving_window_length)
            if local_bxy_remainder_days != 0:
                print('an error in reshaping input data at creating x_train and y_train has occurred')
                print('please recheck before continue')
                return False  # this will return a controlled error,
            local_bxy_day_in_year = []
            [local_bxy_day_in_year.append(local_bxy_last_learning_day_in_year + year * 365) for year in range(local_bxy_nof_years)]

            # building this mild no-triviality 3D arrays for training
            print('defining x_train')
            x_train = []
            if local_bxy_settings_arg['train_model_input_data_approach'] == "all":
                [x_train.append(local_bxy_unit_sales[:, day: day + local_bxy_time_steps_days])
                 for day in range(local_bxy_unit_sales, local_bxy_max_selling_time, local_bxy_stride_window_walk)]
            elif local_bxy_settings_arg['train_model_input_data_approach'] == "focused":
                [x_train.append(local_bxy_unit_sales[:, day: day + local_bxy_moving_window_length])
                 for last_day in local_bxy_day_in_year[:-1]
                 for day in range(last_day + local_bxy_window_output_length,
                                  last_day + local_bxy_window_output_length - local_bxy_days_in_focus_frame, -local_bxy_stride_window_walk)]
                # border condition, take care with last year, working with last data available
                [x_train.append(local_bxy_unit_sales[:, day - local_bxy_moving_window_length: day])
                 for last_day in local_bxy_day_in_year[-1:]
                 for day in range(last_day, last_day - local_bxy_days_in_focus_frame, -local_bxy_stride_window_walk)]
            else:
                logging.info("\ntrain_model_input_data_approach is not defined")
                print('-a problem occurs with the data_approach settings')
                return False, None
            print('defining y_train')
            y_train = []
            if local_bxy_settings_arg['train_model_input_data_approach'] == "all":
                [y_train.append(local_bxy_unit_sales[:,
                                day + local_bxy_stride_window_walk: day + local_bxy_time_steps_days + local_bxy_stride_window_walk])
                 for day in range(local_bxy_time_steps_days, local_bxy_max_selling_time, local_bxy_stride_window_walk)]
            elif local_bxy_settings_arg['train_model_input_data_approach'] == "focused":
                [y_train.append(local_bxy_unit_sales[:,
                                day - local_bxy_stride_window_walk: day + local_bxy_moving_window_length - local_bxy_stride_window_walk])
                 for last_day in local_bxy_day_in_year[:-1]
                 for day in range(last_day + local_bxy_window_output_length,
                                  last_day + local_bxy_window_output_length - local_bxy_days_in_focus_frame, -local_bxy_stride_window_walk)]
                # border condition, take care with last year, working with last data available
                [y_train.append(local_bxy_unit_sales[:,
                                day - local_bxy_moving_window_length - local_bxy_stride_window_walk: day - local_bxy_stride_window_walk])
                 for last_day in local_bxy_day_in_year[-1:]
                 for day in range(last_day, last_day - local_bxy_days_in_focus_frame, -local_bxy_stride_window_walk)]

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            # saving for human-eye review and reassurance; x_train and y_train are 3Ds arrays so...taking the last element
            last_time_step_x_train = x_train[-1:, :, :]
            last_time_step_x_train = last_time_step_x_train.reshape(last_time_step_x_train.shape[1],
                                                                    last_time_step_x_train.shape[2])
            last_time_step_y_train = y_train[-1:, :, :]
            last_time_step_y_train = last_time_step_y_train.reshape(last_time_step_y_train.shape[1],
                                                                    last_time_step_y_train.shape[2])
            np.save(''.join([local_bxy_settings_arg['train_data_path'],
                             'last_time_step_from_external_library_model_x_train']),
                    last_time_step_x_train)
            np.save(''.join([local_bxy_settings_arg['train_data_path'],
                             'last_time_step_from_external_library_y_train']),
                    last_time_step_y_train)
            np.savetxt(''.join([local_bxy_settings_arg['train_data_path'],
                                'last_time_step_from_external_library_x_train.csv']),
                       last_time_step_x_train, fmt='%10.15f', delimiter=',', newline='\n')
            np.savetxt(''.join([local_bxy_settings_arg['train_data_path'],
                                'last_time_step_from_external_library_y_train.csv']),
                       last_time_step_y_train, fmt='%10.15f', delimiter=',', newline='\n')
        except Exception as build_x_y_train_error:
            print('build_x_y_train_ submodule_error: ', build_x_y_train_error)
            logger.info('error at build_x_y_train')
            logger.error(str(build_x_y_train_error), exc_info=True)
            return False
        return x_train, y_train
