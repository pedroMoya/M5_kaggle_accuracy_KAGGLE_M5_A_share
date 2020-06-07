# read and prepare input data, execute model(s) and save forecasts

# importing python libraries and opening settings
try:
    import os
    import sys
    import logging
    import logging.handlers as handlers
    import json
    import datetime
    import numpy as np
    import pandas as pd
    import itertools as it
    import tensorflow as tf
    from tensorflow.keras import backend as kb
    from tensorflow.keras import losses, models
    from tensorflow.keras.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    with open('./settings.json') as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()
    sys.path.insert(1, local_script_settings['custom_library_path'])
    from metaheuristic_module import tuning_metaheuristic
    from model_analyzer import model_structure
    from submission_evaluator import submission_tester
    if local_script_settings['metaheuristic_optimization'] == "True":
        with open(''.join(
                [local_script_settings['metaheuristics_path'], 'organic_settings.json'])) as local_json_file:
            local_script_settings = json.loads(local_json_file.read())
            local_json_file.close()
        metaheuristic_predict = tuning_metaheuristic()
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float32')
except Exception as ee1:
    print('Error importing libraries or opening settings (predict module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# keras session and random seed reset/fix, set epsilon keras backend
kb.clear_session()
np.random.seed(1)
tf.random.set_seed(2)
kb.set_epsilon(1)  # needed while using "mape" as one of the metric at training model


# classes definitions


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


# functions definitions


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


def predict():
    try:
        print('\n~predict module~')
        # from 1th june, here get real unit_sales for days d_1914 to d_1941,
        # for model optimization, but avoiding overfitting

        # open predict settings
        with open(''.join([local_script_settings['test_data_path'], 'forecast_settings.json'])) as local_f_json_file:
            forecast_settings = json.loads(local_f_json_file.read())
            local_f_json_file.close()

        # load clean data (divided in groups) and time_serie_group
        scaled_unit_sales_g1 = np.load(''.join([local_script_settings['train_data_path'], 'group1.npy']),
                                       allow_pickle=True)
        scaled_unit_sales_g2 = np.load(''.join([local_script_settings['train_data_path'], 'group2.npy']),
                                       allow_pickle=True)
        scaled_unit_sales_g3 = np.load(''.join([local_script_settings['train_data_path'], 'group3.npy']),
                                       allow_pickle=True)
        groups_list = [scaled_unit_sales_g1, scaled_unit_sales_g2, scaled_unit_sales_g3]
        time_series_group = np.load(''.join([local_script_settings['train_data_path'], 'time_serie_group.npy']),
                                    allow_pickle=True)
        # print(time_series_group.shape)
        # print(time_series_group)
        # store the number of time_series and max_selling_time of each group
        if local_script_settings['automatic_time_series_number'] == 'True':
            number_of_time_series_g1 = np.shape(scaled_unit_sales_g1)[0]
            number_of_time_series_g2 = np.shape(scaled_unit_sales_g2)[0]
            number_of_time_series_g3 = np.shape(scaled_unit_sales_g3)[0]
        else:
            # open forecast settings
            with open(''.join([local_script_settings['test_data_path'],
                               'forecast_settings.json'])) as local_f_json_file:
                forecast_settings = json.loads(local_f_json_file.read())
                local_f_json_file.close()

        if local_script_settings['model_analyzer'] == 'on':
            analyzer = model_structure()
            # requires hdf5 format
            model_name = '_high_loss_time_serie_model_forecaster_in_block_.h5'
            analysis_result = analyzer.analize(model_name, local_script_settings)
            if analysis_result:
                print('model_analysis successfully, json file saved')
            else:
                print('error at model_analysis submodule')

        # open raw_data
        raw_data_filename = 'sales_train_evaluation.csv'
        raw_data_sales = pd.read_csv(''.join([local_script_settings['raw_data_path'], raw_data_filename]))
        print('raw sales data accessed')

        # extract data and check  dimensions
        raw_unit_sales = raw_data_sales.iloc[:, 6:].values
        max_selling_time = np.shape(raw_unit_sales)[1]
        local_settings_max_selling_time = local_script_settings['max_selling_time']
        if local_settings_max_selling_time < max_selling_time:
            raw_unit_sales = raw_unit_sales[:, :local_settings_max_selling_time]
        elif max_selling_time != local_settings_max_selling_time:
            print("settings doesn't match data dimensions, it must be rechecked before continue(_predict_module)")
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' data dimensions does not match settings']))
            return False
        else:
            if local_script_settings['competition_stage'] != 'submitting_after_June_1th_using_1941days':
                print(''.join(['\x1b[0;2;41m', 'Warning', '\x1b[0m']))
                print('please check: forecast horizon days will be included within training data')
                print('It was expected that the last 28 days were not included..')
                print('to avoid overfitting')
            elif local_script_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                print(''.join(['\x1b[0;2;41m', 'Straight end of the competition', '\x1b[0m']))
            else:
                print('continuing the training, but a mismatch was found within max_selling and forecast_horizon days')
        print('raw data input collected and check of data dimensions passed (train_module)')

        # make forecast --> '28 days future predictions for unit_sales', organized in groups
        forecast_horizon_days = local_script_settings['forecast_horizon_days']
        # populate with the previously normalized right input for forecasting,
        # *here is the code to access the preprocessed data from _1_prepare_data.py*
        # x_test_from_prepare = groups_list[group][:, -forecast_horizon_days:]
        # x_test_from_prepare = x_test_from_prepare.reshape(1, x_test_from_prepare.shape[1],
        #                                                   x_test_from_prepare.shape[0])
        # in order to not carrie dependencies, _3_predict.py module preprocess again from raw data
        # if needed, could be checked with: x_test_from_prepare == x_test  // output --> [[True]] * shape
        nof_groups = local_script_settings['number_of_groups']
        # if only neural_network (Not stochastic_simulation as first approach), ALL time_series are not_improved...
        nof_time_series = raw_unit_sales.shape[0]
        print('number of time_series:', nof_time_series)
        time_series_not_improved = [time_serie for time_serie in range(nof_time_series)]
        if local_script_settings['skip_neural_network_forecaster_in_predict_module'] == "True":
            print('by settings settled, skipping neural_network forecaster')
            print('using only first model for forecasting')
            # reading stochastic simulation forecasts
            all_forecasts = np.load(''.join([local_script_settings['train_data_path'],
                                             'stochastic_simulation_forecasts.npy']))
        else:
            all_forecasts = np.zeros(shape=(nof_time_series * 2, forecast_horizon_days))
            time_steps_days = local_script_settings['time_steps_days']
            if local_script_settings['first_train_approach'] == 'stochastic_simulation':
                nof_groups = 1
                time_series_not_improved = np.load(''.join([local_script_settings['models_evaluation_path'],
                                                            'time_series_not_improved.npy']), allow_pickle=True)
                # managing that only 1 group will be used
                time_series_group = np.array([[time_serie, 0] for time_serie in time_series_not_improved])
                groups_list = [raw_unit_sales]
            for group in range(nof_groups):
                # print(time_series_group.shape)
                time_series_in_group = time_series_group[:, [0]][time_series_group[:, [1]] == group]
                # this commented code is replaced for "the managing that only 1 group will be used" about six line above
                # if local_script_settings['first_train_approach'] == 'stochastic_simulation':
                #     time_series_in_group = time_series_not_improved
                print('time_series group shape, len:', time_series_in_group.shape, len(time_series_in_group))
                x_input = groups_list[group][time_series_in_group, -time_steps_days:]
                x_input = x_input.reshape(1, x_input.shape[1], x_input.shape[0])
                print('x_input shape: ', np.shape(x_input))

                # load model and make forecast for the time serie
                if nof_groups > 1:
                    forecaster = models.load_model(''.join([local_script_settings['models_path'],
                                                            'model_group_', str(group),
                                                            '_forecast_.h5']),
                                                   custom_objects={'modified_mape': modified_mape,
                                                                   'customized_loss': customized_loss})
                    # this full-group model was not obtained better results
                    point_forecast_original = forecaster.predict(x_input)
                    print('forecast shape: ', np.shape(point_forecast_original))
                    print('group: ', group, '\ttime_serie: all ones belonging to this group')

                else:
                    # one model and one group with all the time series, one template for all, but with different weights
                    # forecaster = models.load_model(''.join([local_script_settings['models_path'],
                    #                                         'generic_forecaster_template_individual_ts.h5']),
                    #                                custom_objects={'modified_mape': modified_mape,
                    #                                                'customized_loss': customized_loss})
                    model_name = ''.join(['generic_forecaster_template_individual_ts.json'])
                    json_file = open(''.join([local_script_settings['models_path'], model_name]), 'r')
                    model_json = json_file.read()
                    json_file.close()
                    forecaster = models.model_from_json(model_json)
                    print('model structure loaded')
                    forecaster.summary()
                    for time_serie in time_series_not_improved:
                        # load weights of respective time_serie model
                        print('group: ', group, '\ttime_serie:', time_serie)
                        forecaster.load_weights(''.join([local_script_settings['models_path'],
                                                         '/weights_zero_removed/_individual_ts_',
                                                         str(time_serie), '_model_weights_.h5']))
                        point_forecast_original = forecaster.predict(x_input[:, :, time_serie: time_serie + 1])
                        print('forecast shape: ', np.shape(point_forecast_original))
                        # inverse reshape
                        point_forecast = point_forecast_original.reshape(point_forecast_original.shape[1])
                        point_forecast = point_forecast[-forecast_horizon_days:]
                        print('point_forecast shape:', point_forecast.shape)
                        all_forecasts[time_serie, :] = point_forecast

                    # save points forecast of NN model
                    np.savetxt(''.join([local_script_settings['others_outputs_path'], 'point_forecast_',
                                        '_group_', str(group), '_only_NN_.csv']), point_forecast, fmt='%10.15f',
                               delimiter=',', newline='\n')
                    print('point forecasts saved to file')

                    # inserting zeros as was determinate by first model (stochastic simulation)
                    zero_loc = np.load(''.join([local_script_settings['train_data_path'], 'zero_localizations.npy']),
                                       allow_pickle=True)
                    all_forecasts[zero_loc[:, 0], zero_loc[:, 1]] = 0


                # saving consolidated submission
                submission = np.genfromtxt(''.join([local_script_settings['raw_data_path'], 'sample_submission.csv']),
                                           delimiter=',', dtype=None, encoding=None)
                if local_script_settings['competition_stage'] == 'submitting_after_June_1th_using_1913days':
                    submission[1:, 1:] = all_forecasts
                    submission[30491:, 1:] = 7  # only checking reach that data
                elif local_script_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                    # fill validation rows with the real data used for training, and evaluation rows with the forecasts
                    submission[1:30490, 1:] = raw_unit_sales[:, -forecast_horizon_days]
                    submission[30491:, 1:] = all_forecasts[:30490, -forecast_horizon_days]
                pd.DataFrame(submission).to_csv(''.join([local_script_settings['submission_path'], 'submission.csv']),
                                                index=False, header=None)
                np.savetxt(''.join([local_script_settings['others_outputs_path'],
                                    'point_forecast_ss_and_or_nn_models_applied_.csv']),
                           all_forecasts, fmt='%10.15f', delimiter=',', newline='\n')
                print('forecast saved, submission file built and stores')
                print("forecast subprocess ended successfully")
                logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                     ' correct forecasting process']))
    except Exception as e1:
        print('Error in predict module')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' predict module error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
