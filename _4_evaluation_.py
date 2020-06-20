# evaluation module
# open forecasts done, ground truth data and applies metrics for evaluate models
# save results

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
    from high_loss_identified_ts_forecast_module import individual_high_loss_ts_forecast
    from organic_in_block_high_loss_identified_ts_forecast_module import in_block_high_loss_ts_forecast
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
    print('Error importing libraries or opening settings (evaluation module)')
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


def evaluate():
    try:
        print('\n~evaluation module~')
        # from 1th june, here get real unit_sales for days d_1914 to d_1941,
        # for model optimization, but avoiding overfitting
        # open raw_data
        raw_data_filename = 'sales_train_evaluation.csv'
        raw_data_sales = pd.read_csv(''.join([local_script_settings['raw_data_path'], raw_data_filename]))
        print('raw sales data accessed')

        # extract data and check  dimensions
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
                print('please check: forecast horizon days was included within the training data')
                print('It was expected that the last 28 days were not included..')
                print('to avoid overfitting')
            elif local_script_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                print(''.join(['\x1b[0;2;41m', 'Straight end of the competition', '\x1b[0m']))
        print('raw data input collected and check of data dimensions passed (evaluation_module)')

        if local_script_settings['model_analyzer'] == 'on':
            analyzer = model_structure()
            model_name = 'generic_forecaster_template_individual_ts.h5'
            analysis_result = analyzer.analize(model_name, local_script_settings)
            if analysis_result:
                print('model_analysis successfully, json file saved')
            else:
                print('error at model_analysis submodule')

            # evaluation of models forecasts according to day-wise comparison
            # forecaster(x_test) <=> y_pred
            print('\nmodels evaluation\nusing MEAN SQUARED ERROR, '
                  'MODIFIED-MEAN ABSOLUTE PERCENTAGE ERROR and MEAN ABSOLUTE ERROR')
            print('{:^19s}{:^19s}{:^19s}{:^19s}'.format('time_serie', 'error_metric_MSE',
                                                        'error_metric_Mod_MAPE', 'error_metric_MAPE'))
            time_series_error_mse = []
            time_series_error_mod_mape = []
            time_series_error_mape = []
            y_ground_truth_array = []
            y_pred_array = []
            customized_mod_mape = modified_mape()
            if local_script_settings['first_train_approach'] == 'stochastic_simulation':
                nof_groups = 1
            for group in range(nof_groups):
                time_series_in_group = time_series_group[:, [0]][time_series_group[:, [1]] == group]
                if local_script_settings['first_train_approach'] == 'stochastic_simulation':
                    time_series_in_group = time_series_not_improved
                time_serie_iterator = 0
                for time_serie in time_series_in_group:
                    y_ground_truth = raw_unit_sales[time_serie, -forecast_horizon_days:]
                    y_pred = forecasts[group][time_serie_iterator, -forecast_horizon_days:].flatten()
                    error_metric_mse = mean_squared_error(y_ground_truth, y_pred)
                    error_metric_mod_mape = 100 * customized_mod_mape(y_ground_truth, y_pred)
                    error_metric_mape = mean_absolute_percentage_error(y_ground_truth, y_pred)
                    # print('{:^19d}{:^19f}{:^19f}{:^19f}'.format(time_serie, error_metric_mse,
                    #                                             error_metric_mod_mape, error_metric_mape))
                    time_series_error_mse.append([time_serie, error_metric_mse])
                    time_series_error_mod_mape.append(error_metric_mod_mape)
                    time_series_error_mape.append(error_metric_mape)
                    y_ground_truth_array.append(y_ground_truth)
                    y_pred_array.append(y_pred)
                    time_serie_iterator += 1
                print('model evaluation subprocess ended successfully')
                mean_mse = np.mean([loss_mse[1] for loss_mse in time_series_error_mse])
                print('time_serie mean mse: ', mean_mse)
                if local_script_settings['metaheuristic_optimization'] == "True":
                    model_evaluation = metaheuristic_predict.evaluation_brain(mean_mse, local_script_settings)
                    if model_evaluation[0] and not model_evaluation[1]:
                        print('model evaluated did not get better results than previous ones')
                    elif not model_evaluation[0]:
                        print('error in meta_heuristic evaluation submodule')
                    else:
                        print('model evaluated got better results than previous ones')

            # treating time series with mediocre to bad forecasts (high loss) calling the specific submodule
            if local_script_settings['repeat_training_in_block'] == "True" \
                    and local_script_settings['first_train_approach'] != 'stochastic_simulation':
                in_block_time_series_forecast = in_block_high_loss_ts_forecast()
                time_series_reviewed = in_block_time_series_forecast.forecast(local_settings=local_script_settings,
                                                                              local_raw_unit_sales=raw_unit_sales,
                                                                              local_mse=time_series_error_mse)
                print('last step -time_serie specific (in-block) forecast- completed, success: ', time_series_reviewed)

        # evaluate external o internal csv file submission (in 9.3_OTHERS_INPUTS folder)
        if local_script_settings['submission_evaluation'] == "True":
            submission_evaluation = submission_tester()
            external_submission_reviewed = submission_evaluation.evaluate_external_submit(
                forecast_horizon_days, local_settings=local_script_settings)
            internal_submission_reviewed = submission_evaluation.evaluate_internal_submit(
                forecast_horizon_days, all_forecasts, local_settings=local_script_settings)
            print('evaluation of external submission forecast- completed, success: ',
                  external_submission_reviewed)
            print('evaluation of internal submission forecast- completed, success: ',
                  internal_submission_reviewed)

    except Exception as e1:
        print('Error in evaluator module')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' evaluator module error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
