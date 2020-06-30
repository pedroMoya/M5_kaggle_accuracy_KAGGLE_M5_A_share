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
    from model_analyzer import model_structure
    from submission_evaluator import submission_tester
    from explore_results_mse import explore_results_and_generate_submission
    from day_by_day_best_low_error_point_forecast_between_all_models \
        import explore_day_by_day_results_and_generate_submission
    from build_forecast_focused_in_results import explore_results_focused_and_generate_submission

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


# functions definitions


def cof_zeros(array, local_cof_settings):
    if local_cof_settings['zeros_control'] == "True":
        local_max = np.amax(array) + 1
        array[array <= 0] = local_max
        local_min = np.amin(array)
        array[array == local_max] = local_min
    return array


def evaluate():
    try:
        print('\n~evaluation module~')
        if local_script_settings['evaluate_individual_ts_LSTM'] == 'True':
            # from 1th june, here get real unit_sales for days d_1914 to d_1941,
            # for model optimization, but remember avoiding overfitting
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
                    print('please check: forecast horizon days was included within the training data')
                    print('It was expected that the last 28 days were not included..')
                    print('to avoid overfitting')
                elif local_script_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                    print(''.join(['\x1b[0;2;41m', 'Straight end of the competition', '\x1b[0m']))
            print('raw data input collected and check of data dimensions passed (evaluation_module)')

            # evaluation of models forecasts according to day-wise comparison
            # forecaster(x_test) <=> y_pred
            print('\nmodels evaluation\nusing MEAN SQUARED ERROR')
            time_series_error_mse = []
            y_pred_array = []

            time_serie_not_improved_with_first_model = \
                np.load(''.join([local_script_settings['models_evaluation_path'],
                                 'time_series_not_improved_stochastic_simulation.npy']))
            forecast_horizon_days = local_script_settings['forecast_horizon_days']
            nof_time_series = np.shape(raw_unit_sales)[0]
            current_trained_model_subdirectory = local_script_settings['current_trained_model_subdirectory']
            path_to_model = ''.join([local_script_settings['models_path'], current_trained_model_subdirectory])
            time_serie_iterator = 0
            lstm_forecast = []
            current_model_name = local_script_settings['current_model_name']
            forecaster = models.load_model(''.join([local_script_settings['models_path'], current_model_name]))
            model_weights_name_template = local_script_settings['model_weights_name_template']
            dummy_data = np.array([0] * forecast_horizon_days)
            for time_serie in range(nof_time_series):
                model_weights_name = model_weights_name_template.replace('x', str(time_serie))
                model_full_path_filename = ''.join([path_to_model, model_weights_name])
                # check if trained model exist
                if os.path.isfile(model_full_path_filename):
                    y_ground_truth = raw_unit_sales[time_serie, -forecast_horizon_days:]
                    x_input = raw_unit_sales[time_serie: time_serie + 1, -2 * forecast_horizon_days: -forecast_horizon_days]
                    x_input = cof_zeros(x_input, local_script_settings)
                    x_input = x_input.reshape(1, x_input.shape[1], 1).astype(np.dtype('float32'))
                    forecaster.load_weights(model_full_path_filename)
                    y_pred = forecaster.predict(x_input)
                    y_pred = y_pred.reshape(y_pred.shape[1])
                    y_pred = cof_zeros(y_pred, local_script_settings)
                    error_metric_mse = mean_squared_error(y_ground_truth, y_pred)
                    time_series_error_mse.append([time_serie, error_metric_mse])
                    y_pred_array.append(y_pred)
                    time_serie_iterator += 1
                    lstm_forecast.append([time_serie, error_metric_mse])
                else:
                    y_pred_array.append(dummy_data)  # same as below, necessary for full time_serie, but is don't used
                    lstm_forecast.append([time_serie, 1000])  # necessarily a number, meaning "no model for this time serie"
            lstm_forecast = np.array(lstm_forecast)
            mean_mse = np.mean([loss_mse[1] for loss_mse in time_series_error_mse])
            print('time_serie mean mse: ', mean_mse)
            np.save(''.join([local_script_settings['models_evaluation_path'], 'LSTM_model_forecast_result_mse']),
                    lstm_forecast)
            np.savetxt(''.join([local_script_settings['models_evaluation_path'], 'LSTM_model_forecast_result_mse.csv']),
                       lstm_forecast, fmt='%10.15f', delimiter=',', newline='\n')
            np.save(''.join([local_script_settings['train_data_path'], 'LSTM_model_forecast_data']),
                    y_pred_array)
            np.savetxt(''.join([local_script_settings['train_data_path'], 'LSTM_model_forecast_data.csv']),
                       y_pred_array, fmt='%10.15f', delimiter=',', newline='\n')
            np.save(''.join([local_script_settings['models_evaluation_path'], 'LSTM_ts_error_MSE_']), time_series_error_mse)
            np.savetxt(''.join([local_script_settings['models_evaluation_path'], 'LSTM_ts_error_MSE_.csv']),
                       time_series_error_mse, fmt='%10.15f', delimiter=',', newline='\n')
            print('models evaluation metrics for each time serie forecast saved to file')
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' successful saved error metrics']))

        # generate diagram of neural network model
        if local_script_settings['model_analyzer'] == 'on':
            analyzer = model_structure()
            model_name = '_acc_freq_in_block_nn_model_.h5'
            analysis_result = analyzer.analize(model_name, local_script_settings)
            if analysis_result:
                print('model_analysis successfully, json file saved')
            else:
                print('error at model_analysis submodule')

        # calling submodule that obtain the best forecast for each time_serie between various models
        # (by time_series, days in block)
        # building one best SUBMISSION for this approach
        explore_results_and_generate_submission_engine = explore_results_and_generate_submission()
        explore_results_and_generate_submission_review = explore_results_and_generate_submission_engine.run(
            'mse_based_best_ts_forecast', local_script_settings)
        if explore_results_and_generate_submission_review:
            print('mse best submission between models obtained')
        else:
            print('an error has occurred in generating between different-models best forecasts submission')

        # calling submodule that obtain the best forecast for each time_serie between various models day_by_day
        # (by time_series, and day by day)
        # building one best SUBMISSION for this approach
        # explore_day_by_day_results_and_generate_submission_engine = explore_day_by_day_results_and_generate_submission()
        # explore_day_by_day_results_and_generate_submission_review = \
        #     explore_day_by_day_results_and_generate_submission_engine.run(
        #         'day_by_day_based_best_lower_error_ts_forecast', local_script_settings)
        # if explore_day_by_day_results_and_generate_submission_review:
        #     print('lower point_forecast error best submission between models '
        #           'in a ts by ts and day by day based approach obtained')
        # else:
        #     print('an error has occurred in generating ts by ts and day by day based approach '
        #           'best forecasts submission')

        # calling submodule that make best_mse forecast for each time_serie between various models
        # according to improved or not, select distribution
        # building one best SUBMISSION for this approach
        explore_results_and_generate_submission_engine = explore_results_focused_and_generate_submission()
        explore_results_and_generate_submission_review = \
            explore_results_and_generate_submission_engine.run(
                'best_mse_and_distribution_model_forecast', local_script_settings)
        if explore_results_and_generate_submission_review:
            print('mse_results_and_distribution best submission between models or stochastic approach')
        else:
            print('an error has occurred in mse_results_and_distribution best submission '
                  'between models or stochastic approach')

        # # finalizing the last module
        print('model evaluation subprocess ended successfully')
    except Exception as e1:
        print('Error in evaluator module')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' evaluator module error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
