# High loss identified time series module
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

# set random seed for reproducibility --> done in _2_train.py module
# np.random.seed(3)

# clear session
kb.clear_session()


def general_mean_scaler(local_array):
    if len(local_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_array, axis=1)
    mean_scaling = np.divide(local_array, 1 + mean_local_array)
    return mean_scaling, mean_local_array


def window_based_normalizer(local_window_array):
    if len(local_window_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_window_array, axis=1)
    window_based_normalized_array = np.add(local_window_array, -mean_local_array)
    return window_based_normalized_array, mean_local_array


def general_mean_rescaler(local_array, local_complete_array_unit_mean, local_forecast_horizon):
    if len(local_array) == 0:
        return "argument length 0"
    local_array = local_array.clip(0)
    local_complete_array_unit_mean = np.array([local_complete_array_unit_mean, ] * local_forecast_horizon).transpose()
    mean_rescaling = np.multiply(local_array, 1 + local_complete_array_unit_mean)
    return mean_rescaling


def window_based_denormalizer(local_window_array, local_last_window_mean, local_forecast_horizon):
    if len(local_window_array) == 0:
        return "argument length 0"
    local_last_window_mean = np.array([local_last_window_mean, ] * local_forecast_horizon).transpose()
    window_based_denormalized_array = np.add(local_window_array, local_last_window_mean)
    return window_based_denormalized_array


class modified_mape(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.cast(local_true, dtype=tf.float32)
        local_pred = tf.cast(local_pred, dtype=tf.float32)
        numerator = tf.abs(tf.add(local_pred, -local_true))
        denominator = tf.add(tf.convert_to_tensor(1., dtype=tf.float32), tf.abs(local_true))
        return tf.math.divide_no_nan(numerator, denominator)


class customized_loss(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.convert_to_tensor(local_true, dtype=tf.float32)
        local_pred = tf.convert_to_tensor(local_pred, dtype=tf.float32)
        factor_difference = tf.reduce_mean(tf.abs(tf.add(local_pred, -local_true)))
        factor_true = tf.reduce_mean(tf.add(tf.convert_to_tensor(1., dtype=tf.float32), local_true))
        return tf.math.multiply_no_nan(factor_difference, factor_true)


class in_block_high_loss_ts_forecast:

    def forecast(self, local_mse, local_normalized_scaled_unit_sales, 
                 local_mean_unit_complete_time_serie, local_raw_unit_sales, local_settings):
        try:
            print('starting high loss (mse in previous LSTM) time_series in-block forecast submodule')
            # set training parameters
            with open(''.join([local_settings['hyperparameters_path'],
                               'in_block_time_serie_based_model_hyperparameters.json'])) \
                    as local_r_json_file:
                model_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            local_time_series_group = np.load(''.join([local_settings['train_data_path'], 'time_serie_group.npy']),
                                              allow_pickle=True)
            time_steps_days = int(local_settings['time_steps_days'])
            epochs = int(model_hyperparameters['epochs'])
            batch_size = int(model_hyperparameters['batch_size'])
            workers = int(model_hyperparameters['workers'])
            optimizer_function = model_hyperparameters['optimizer']
            optimizer_learning_rate = model_hyperparameters['learning_rate']
            if optimizer_function == 'adam':
                optimizer_function = optimizers.Adam(optimizer_learning_rate)
            elif optimizer_function == 'ftrl':
                optimizer_function = optimizers.Ftrl(optimizer_learning_rate)
            losses_list = []
            loss_1 = model_hyperparameters['loss_1']
            loss_2 = model_hyperparameters['loss_2']
            loss_3 = model_hyperparameters['loss_3']
            union_settings_losses = [loss_1, loss_2, loss_3]
            if 'mape' in union_settings_losses:
                losses_list.append(losses.MeanAbsolutePercentageError())
            if 'mse' in union_settings_losses:
                losses_list.append(losses.MeanSquaredError())
            if 'mae' in union_settings_losses:
                losses_list.append(losses.MeanAbsoluteError())
            if 'm_mape' in union_settings_losses:
                losses_list.append(modified_mape())
            if 'customized_loss_function' in union_settings_losses:
                losses_list.append(customized_loss())
            metrics_list = []
            metric1 = model_hyperparameters['metrics1']
            metric2 = model_hyperparameters['metrics2']
            union_settings_metrics = [metric1, metric2]
            if 'rmse' in union_settings_metrics:
                metrics_list.append(metrics.RootMeanSquaredError())
            if 'mse' in union_settings_metrics:
                metrics_list.append(metrics.MeanSquaredError())
            if 'mae' in union_settings_metrics:
                metrics_list.append(metrics.MeanAbsoluteError())
            if 'mape' in union_settings_metrics:
                metrics_list.append(metrics.MeanAbsolutePercentageError())
            l1 = model_hyperparameters['l1']
            l2 = model_hyperparameters['l2']
            if model_hyperparameters['regularizers_l1_l2'] == 'True':
                activation_regularizer = regularizers.l1_l2(l1=l1, l2=l2)
            else:
                activation_regularizer = None

            # searching for time_series with high loss forecast
            time_series_treated = []
            poor_results_mse_threshold = local_settings['poor_results_mse_threshold']
            poor_result_time_serie_list = []
            nof_features_for_training = 0
            for result in local_mse:
                if result[1] > poor_results_mse_threshold:
                    nof_features_for_training += 1
                    poor_result_time_serie_list.append(int(result[0]))
            # nof_features_for_training = local_normalized_scaled_unit_sales.shape[0]
            nof_features_for_training = len(poor_result_time_serie_list)
            # creating model
            forecaster_in_block = tf.keras.Sequential()
            print('current model for specific high loss time_series: Mix_Bid_PeepHole_LSTM_Dense_ANN')
            # first layer (DENSE)
            if model_hyperparameters['units_layer_1'] > 0:
                forecaster_in_block.add(layers.Dense(units=model_hyperparameters['units_layer_1'],
                                                     activation=model_hyperparameters['activation_1'],
                                                     input_shape=(model_hyperparameters['time_steps_days'],
                                                                  nof_features_for_training),
                                                     activity_regularizer=activation_regularizer))
                forecaster_in_block.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_1'])))
            # second LSTM layer
            if model_hyperparameters['units_layer_2'] > 0:
                forecaster_in_block.add(layers.Bidirectional(layers.RNN(
                    PeepholeLSTMCell(units=model_hyperparameters['units_layer_2'],
                                     activation=model_hyperparameters['activation_2'],
                                     activity_regularizer=activation_regularizer,
                                     dropout=float(model_hyperparameters['dropout_layer_2'])),
                    return_sequences=False)))
                forecaster_in_block.add(RepeatVector(model_hyperparameters['repeat_vector']))
            # third LSTM layer
            if model_hyperparameters['units_layer_3'] > 0:
                forecaster_in_block.add(layers.Bidirectional(layers.RNN(
                    PeepholeLSTMCell(units=model_hyperparameters['units_layer_3'],
                                     activation=model_hyperparameters['activation_3'],
                                     activity_regularizer=activation_regularizer,
                                     dropout=float(model_hyperparameters['dropout_layer_3'])),
                    return_sequences=False)))
                forecaster_in_block.add(RepeatVector(model_hyperparameters['repeat_vector']))
            # fourth layer (DENSE)
            if model_hyperparameters['units_layer_4'] > 0:
                forecaster_in_block.add(layers.Dense(units=model_hyperparameters['units_layer_4'],
                                                     activation=model_hyperparameters['activation_4'],
                                                     activity_regularizer=activation_regularizer))
                forecaster_in_block.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_4'])))
            # final layer
            forecaster_in_block.add(TimeDistributed(layers.Dense(units=nof_features_for_training)))
            # forecaster_in_block.saves(''.join([local_settings['models_path'], '_model_structure_']),
            #                 save_format='tf')
            forecast_horizon_days = local_settings['forecast_horizon_days']
            forecaster_in_block.build(input_shape=(1, forecast_horizon_days, nof_features_for_training))
            forecaster_in_block.compile(optimizer=optimizer_function,
                                        loss=losses_list,
                                        metrics=metrics_list)
            forecaster_in_block_json = forecaster_in_block.to_json()
            with open(''.join([local_settings['models_path'], 'forecaster_in_block.json']), 'w') as json_file:
                json_file.write(forecaster_in_block_json)
                json_file.close()
            forecaster_in_block_untrained = forecaster_in_block
            print('specific time_serie model initialized and compiled')
            nof_selling_days = local_normalized_scaled_unit_sales.shape[1]
            last_learning_day_in_year = np.mod(nof_selling_days, 365)
            max_selling_time = local_settings['max_selling_time']
            days_in_focus_frame = model_hyperparameters['days_in_focus_frame']
            window_input_length = local_settings['moving_window_input_length']
            window_output_length = local_settings['moving_window_output_length']
            moving_window_length = window_input_length + window_output_length
            nof_years = local_settings['number_of_years_ceil']

            # training
            # time_serie_data = local_normalized_scaled_unit_sales
            nof_poor_result_time_series = len(poor_result_time_serie_list)
            time_serie_data = np.zeros(shape=(nof_poor_result_time_series, max_selling_time))
            time_serie_iterator = 0
            for time_serie in poor_result_time_serie_list:
                time_serie_data[time_serie_iterator, :] = local_normalized_scaled_unit_sales[time_serie, :]
                time_serie_iterator += 1
            if local_settings['repeat_training_in_block'] == "True":
                print('starting in-block training of model for high_loss time_series in previous model')
                nof_selling_days = time_serie_data.shape[1]
                # nof_moving_windows = np.int32(nof_selling_days / moving_window_length)
                remainder_days = np.mod(nof_selling_days, moving_window_length)
                window_first_days = [first_day
                                     for first_day in range(0, nof_selling_days, moving_window_length)]
                length_window_walk = len(window_first_days)
                # last_window_start = window_first_days[length_window_walk - 1]
                if remainder_days != 0:
                    window_first_days[length_window_walk - 1] = nof_selling_days - moving_window_length
                day_in_year = []
                [day_in_year.append(last_learning_day_in_year + year * 365) for year in range(nof_years)]
                stride_window_walk = model_hyperparameters['stride_window_walk']
                print('defining x_train')
                x_train = []
                if local_settings['train_model_input_data_approach'] == "all":
                    [x_train.append(time_serie_data[:, day - time_steps_days: day - window_output_length])
                     for day in range(time_steps_days, max_selling_time, stride_window_walk)]
                elif local_settings['train_model_input_data_approach'] == "focused":
                    [x_train.append(time_serie_data[:, day: day + time_steps_days])
                     for last_day in day_in_year[:-1]
                     for day in range(last_day + window_output_length,
                                      last_day + window_output_length - days_in_focus_frame, -stride_window_walk)]
                    # border condition, take care with last year, working with last data available, yeah really!!
                    [x_train.append(np.concatenate(
                        (time_serie_data[:, day - window_output_length: day],
                         np.zeros(shape=(nof_poor_result_time_series, time_steps_days - window_output_length))),
                        axis=1))
                     for last_day in day_in_year[-1:]
                     for day in range(last_day, last_day - days_in_focus_frame, -stride_window_walk)]
                else:
                    logging.info("\ntrain_model_input_data_approach is not defined")
                    print('-a problem occurs with the data_approach settings')
                    return False, None
                print('defining y_train')
                y_train = []
                if local_settings['train_model_input_data_approach'] == "all":
                    [y_train.append(time_serie_data[:, day - time_steps_days: day])
                     for day in range(time_steps_days, max_selling_time, stride_window_walk)]
                elif local_settings['train_model_input_data_approach'] == "focused":
                    [y_train.append(time_serie_data[:, day: day + time_steps_days])
                     for last_day in day_in_year[:-1]
                     for day in range(last_day + window_output_length,
                                      last_day + window_output_length - days_in_focus_frame, -stride_window_walk)]
                    # border condition, take care with last year, working with last data available, yeah really!!
                    [y_train.append(np.concatenate(
                        (time_serie_data[:, day - window_output_length: day],
                         np.zeros(shape=(nof_poor_result_time_series, time_steps_days - window_output_length))),
                        axis=1))
                     for last_day in day_in_year[-1:]
                     for day in range(last_day, last_day - days_in_focus_frame, -stride_window_walk)]

                # if time_enhance is active, assigns more weight to the last time_steps according to enhance_last_stride
                if local_settings['time_enhance'] == 'True':
                    enhance_last_stride = local_settings['enhance_last_stride']
                    last_elements = []
                    length_x_y_train = len(x_train)
                    x_train_enhanced, y_train_enhanced = [[]] * 2
                    enhance_iterator = 1
                    for position in range(length_x_y_train - enhance_last_stride, length_x_y_train, -1):
                        [x_train_enhanced.append(x_train[position]) for enhance in range(1, 3 * (enhance_iterator + 1))]
                        [y_train_enhanced.append(y_train[position]) for enhance in range(1, 3 * (enhance_iterator + 1))]
                        enhance_iterator += 1
                    x_train = x_train[:-enhance_last_stride]
                    [x_train.append(time_step) for time_step in x_train_enhanced]
                    y_train = y_train[:-enhance_last_stride]
                    [y_train.append(time_step) for time_step in y_train_enhanced]

                # broadcasts lists to np arrays and applies the last pre-training preprocessing (amplification)
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                print('x_train_shape:  ', x_train.shape)
                if local_settings['amplification'] == 'True':
                    factor = local_settings['amplification_factor']  # factor tuning was done previously
                    for time_serie_iterator in range(np.shape(x_train)[1]):
                        max_time_serie = np.amax(x_train[:, time_serie_iterator, :])
                        x_train[:, time_serie_iterator, :][x_train[:, time_serie_iterator, :] > 0] = \
                            max_time_serie * factor
                        max_time_serie = np.amax(y_train[:, time_serie_iterator, :])
                        y_train[:, time_serie_iterator, :][y_train[:, time_serie_iterator, :] > 0] = \
                            max_time_serie * factor
                print('x_train and y_train built done')

                # define callbacks, checkpoints namepaths
                model_weights = ''.join([local_settings['checkpoints_path'],
                                         'check_point_model_for_high_loss_time_serie_',
                                         model_hyperparameters['current_model_name'],
                                         "_loss_-{loss:.4f}-.hdf5"])
                callback1 = cb.EarlyStopping(monitor='loss', patience=model_hyperparameters['early_stopping_patience'])
                callback2 = cb.ModelCheckpoint(model_weights, monitor='loss', verbose=1,
                                               save_best_only=True, mode='min')
                callbacks = [callback1, callback2]
                x_train = x_train.reshape((np.shape(x_train)[0], np.shape(x_train)[2], np.shape(x_train)[1]))
                y_train = y_train.reshape((np.shape(y_train)[0], np.shape(y_train)[2], np.shape(y_train)[1]))
                print('input_shape: ', np.shape(x_train))

                # train for each time_serie
                # check settings for repeat or not the training
                forecaster_in_block.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, workers=workers,
                                        callbacks=callbacks, shuffle=False)
                # print summary (informative; but if says "shape = multiple", probably useless)
                forecaster_in_block.summary()
                forecaster_in_block.save(''.join([local_settings['models_path'],
                                                  '_high_loss_time_serie_model_forecaster_in_block_.h5']))
                forecaster_in_block.save_weights(''.join([local_settings['models_path'],
                                                          '_weights_high_loss_ts_model_forecaster_in_block_.h5']))
                print('high loss time_series model trained and saved in hdf5 format .h5')
            else:
                forecaster_in_block.load_weights(''.join([local_settings['models_path'],
                                                          '_weights_high_loss_ts_model_forecaster_in_block_.h5']))
                # forecaster_in_block = models.load_model(''.join([local_settings['models_path'],
                #                                                  '_high_loss_time_serie_model_forecaster_.h5']))
                print('weights of previously trained model loaded')

            # compile model and make forecast (not necessary)
            # forecaster_in_block.compile(optimizer='adam', loss='mse')

            # evaluating model and comparing with aggregated (in-block) LSTM
            print('evaluating the model trained..')
            time_serie_data = time_serie_data.reshape((1, time_serie_data.shape[1], time_serie_data.shape[0]))
            x_input = time_serie_data[:, -forecast_horizon_days:, :]
            y_pred_normalized = forecaster_in_block.predict(x_input)
            # print('output shape: ', y_pred_normalized.shape)
            time_serie_data = time_serie_data.reshape((time_serie_data.shape[2], time_serie_data.shape[1]))
            # print('time_serie data shape: ', np.shape(time_serie_data))
            time_serie_iterator = 0
            improved_time_series_forecast = []
            time_series_not_improved = []
            improved_mse = []
            for time_serie in poor_result_time_serie_list:
                # for time_serie in range(local_normalized_scaled_unit_sales.shape[0]):
                y_truth = local_raw_unit_sales[time_serie: time_serie + 1, -forecast_horizon_days:]
                # print('y_truth shape:', y_truth.shape)

                # reversing preprocess: rescale, denormalize, reshape
                # inverse reshape
                y_pred_reshaped = y_pred_normalized.reshape((y_pred_normalized.shape[2],
                                                             y_pred_normalized.shape[1]))
                y_pred_reshaped = y_pred_reshaped[time_serie_iterator: time_serie_iterator + 1, :]
                # print('y_pred_reshaped shape:', y_pred_reshaped.shape)

                # inverse transform (first moving_windows denormalizing and then general rescaling)
                time_serie_normalized_window_mean = np.mean(time_serie_data[time_serie_iterator,
                                                            -moving_window_length:])
                # print('mean of this time serie (normalized values): ', time_serie_normalized_window_mean)
                local_denormalized_array = window_based_denormalizer(y_pred_reshaped,
                                                                     time_serie_normalized_window_mean,
                                                                     forecast_horizon_days)
                local_point_forecast = general_mean_rescaler(local_denormalized_array,
                                                             local_mean_unit_complete_time_serie[time_serie],
                                                             forecast_horizon_days)
                # print('rescaled denormalized forecasts array shape: ', local_point_forecast.shape)

                # calculating MSE
                # print(y_truth.shape)
                # print(local_point_forecast.shape)
                local_error_metric_mse = mean_squared_error(y_truth, local_point_forecast)
                # print('time_serie: ', time_serie, '\tMean_Squared_Error: ', local_error_metric_mse)
                previous_result = local_mse[:, 1][local_mse[:, 0] == time_serie].item()
                time_series_treated.append([int(time_serie), previous_result, local_error_metric_mse])
                if local_error_metric_mse < previous_result:
                    # print('better results with time_serie specific model training')
                    print(time_serie, 'MSE improved from ', previous_result, 'to ', local_error_metric_mse)
                    improved_time_series_forecast.append(int(time_serie))
                    improved_mse.append(local_error_metric_mse)
                else:
                    # print('no better results with time serie specific model training')
                    # print('MSE not improved from: ', previous_result, '\t current mse: ', local_error_metric_mse)
                    time_series_not_improved.append(int(time_serie))
                time_serie_iterator += 1
            time_series_treated = np.array(time_series_treated)
            improved_mse = np.array(improved_mse)
            average_mse_in_block_forecast = np.mean(time_series_treated[:, 2])
            average_mse_improved_ts = np.mean(improved_mse)
            print('poor result time serie list len:', len(poor_result_time_serie_list))
            print('mean_mse for in-block forecast:', average_mse_in_block_forecast)
            print('number of time series with better results with this forecast: ', len(improved_time_series_forecast))
            print('mean_mse of time series with better results with this forecast: ', average_mse_improved_ts)
            print('not improved time series =', len(time_series_not_improved))
            time_series_treated = np.array(time_series_treated)
            improved_time_series_forecast = np.array(improved_time_series_forecast)
            time_series_not_improved = np.array(time_series_not_improved)
            poor_result_time_serie_array = np.array(poor_result_time_serie_list)
            # store data of (individual-approach) time_series forecast successfully improved and those that not
            np.save(''.join([local_settings['models_evaluation_path'], 'poor_result_time_serie_array']),
                    poor_result_time_serie_array)
            np.save(''.join([local_settings['models_evaluation_path'], 'time_series_forecast_results']),
                    time_series_treated)
            np.save(''.join([local_settings['models_evaluation_path'], 'improved_time_series_forecast']),
                    improved_time_series_forecast)
            np.save(''.join([local_settings['models_evaluation_path'], 'time_series_not_improved']),
                    time_series_not_improved)
            np.savetxt(''.join([local_settings['models_evaluation_path'], 'time_series_forecast_results.csv']),
                       time_series_treated, fmt='%10.15f', delimiter=',', newline='\n')
            forecaster_in_block_json = forecaster_in_block.to_json()
            with open(''.join([local_settings['models_path'], 'high_loss_time_serie_model_forecaster_in_block.json']), 'w') \
                    as json_file:
                json_file.write(forecaster_in_block_json)
                json_file.close()
            print('trained model weights and architecture saved')
            print('metadata (results, time_serie with high loss) saved')
            print('forecast improvement done. (high loss time_serie focused) submodule has finished')
        except Exception as submodule_error:
            print('time_series in-block forecast submodule_error: ', submodule_error)
            logger.info('error in forecast of in-block time_series (high_loss_identified_ts_forecast submodule)')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
