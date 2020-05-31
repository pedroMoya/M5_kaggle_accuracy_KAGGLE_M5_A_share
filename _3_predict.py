# open clean data, conform data structures
# training and saving models

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
        # for model optimization, but avoiding without overfitting

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
        #     number_of_time_series_g1 = forecast_settings['number_of_time_series_g1']
        #     number_of_time_series_g2 = forecast_settings['number_of_time_series_g2']
        #     number_of_time_series_g3 = forecast_settings['number_of_time_series_g3']
        # nof_time_series_list = [number_of_time_series_g1, number_of_time_series_g2, number_of_time_series_g3]
        # lines before not used, yet
        # max_selling_time_g1 = np.shape(scaled_unit_sales_g1)[1]
        # max_selling_time_g2 = np.shape(scaled_unit_sales_g2)[1]
        # max_selling_time_g3 = np.shape(scaled_unit_sales_g3)[1]
        # max_selling_time_list = [max_selling_time_g1, max_selling_time_g2, max_selling_time_g3]

        # execute model_analysis submodule: input: a h5 format full model, output: architecture in JSON format
        # export a png image of the model plot
        #
        if local_script_settings['model_analyzer'] == 'on':
            analyzer = model_structure()
            model_name = '_high_loss_time_serie_model_forecaster_in_block_.h5'
            analysis_result = analyzer.analize(model_name, local_script_settings)
            if analysis_result:
                print('model_analysis successfully, json file saved')
            else:
                print('error at model_analysis submodule')

        # open raw_data
        raw_data_sales = pd.read_csv(''.join([local_script_settings['raw_data_path'],
                                              'sales_train_validation.csv']))
        print('raw sales data accessed')

        # extract data and check  dimensions
        time_steps_days = local_script_settings['time_steps_days']
        raw_unit_sales = raw_data_sales.iloc[:, 6:].values
        # raw_unit_sales_full = raw_unit_sales
        max_selling_time = np.shape(raw_unit_sales)[1]
        local_settings_max_selling_time = local_script_settings['max_selling_time']
        if local_settings_max_selling_time < max_selling_time:
            raw_unit_sales = raw_unit_sales[:, :local_settings_max_selling_time]
        elif max_selling_time != local_settings_max_selling_time:
            print("settings doesn't match data dimensions, it must be rechecked")
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' data dimensions does not match settings']))
            return False
        else:
            print('warning, please check: forecast horizon is included within training data')
        print('raw data input collected and check of data dimensions passed')

        # data general_mean based - rescaling
        # preproccessing exactly according the way _1_prepare_data algorithm does
        print('preproccessing data, -preprocess must include all data for computing-')
        print('in the same way that was prepared before training')
        nof_time_series = raw_unit_sales.shape[0]
        nof_selling_days = raw_unit_sales.shape[1]
        mean_unit_complete_time_serie = []
        scaled_unit_sales = np.zeros(shape=(nof_time_series, nof_selling_days))
        for time_serie in range(nof_time_series):
            scaled_time_serie = general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[0]
            mean_unit_complete_time_serie.append(general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])[1])
            scaled_unit_sales[time_serie: time_serie + 1, :] = scaled_time_serie
        mean_unit_complete_time_serie = np.array(mean_unit_complete_time_serie)
        print('shape of the preprocessed data array:', np.shape(scaled_unit_sales))
        print('successful rescaling of unit_sale data')

        # data normalization based in moving window
        window_input_length = local_script_settings['moving_window_input_length']
        window_output_length = local_script_settings['moving_window_output_length']
        moving_window_length = window_input_length + window_output_length
        # nof_moving_windows = np.int32(nof_selling_days / moving_window_length)
        remainder_days = np.mod(nof_selling_days, moving_window_length)
        window_first_days = [first_day
                             for first_day in range(0, nof_selling_days, moving_window_length)]
        length_window_walk = len(window_first_days)
        last_window_start = window_first_days[length_window_walk - 1]
        if remainder_days != 0:
            window_first_days[length_window_walk - 1] = nof_selling_days - moving_window_length
        window_normalized_scaled_unit_sales = np.zeros(shape=(nof_time_series, nof_selling_days))
        mean_scaled_window_time_serie = []
        for time_serie in range(nof_time_series):
            normalized_time_serie = []
            for window_start_day in window_first_days:
                window_array = scaled_unit_sales[
                               time_serie: time_serie + 1,
                               window_start_day: window_start_day + moving_window_length]
                normalized_window_array = window_based_normalizer(window_array)[0]
                mean_scaled_window_time_serie.append(window_based_normalizer(window_array)[1])
                if window_start_day == last_window_start:
                    normalized_time_serie.append(normalized_window_array[-remainder_days:])
                else:
                    normalized_time_serie.append(normalized_window_array)
            exact_length_time_serie = np.array(normalized_time_serie).flatten()[: nof_selling_days]
            window_normalized_scaled_unit_sales[time_serie: time_serie + 1, :] = exact_length_time_serie
        mean_scaled_window_time_serie = np.array(mean_scaled_window_time_serie)
        print('input data normalization done')
        print('last preproccesing step finished (..ready for make forecasts..)')

        # make forecast --> '28 days future predictions for unit_sales', organized in groups
        forecast_horizon_days = local_script_settings['forecast_horizon_days']
        # populate with the previously normalized right input for forecasting,
        # *here is the code to access the preprocessed data from _1_prepare_data.py*
        # x_test_from_prepare = groups_list[group][:, -forecast_horizon_days:]
        # x_test_from_prepare = x_test_from_prepare.reshape(1, x_test_from_prepare.shape[1],
        #                                                   x_test_from_prepare.shape[0])
        # in order to not carrie dependencies, _3_predict.py module preprocess again from raw data
        # if needed, could be checked that x_test_from_prepare == x_test --> [[True]] * shape
        nof_groups = local_script_settings['number_of_groups']
        forecasts = []
        for group in range(nof_groups):
            # print(time_series_group.shape)
            time_series_in_group = time_series_group[:, [0]][time_series_group[:, [1]] == group]
            # print(time_series_in_group.shape)
            # print(time_series_in_group)
            x_test = window_normalized_scaled_unit_sales[
                     [time_serie for time_serie in time_series_in_group], -time_steps_days:]
            x_test = x_test.reshape(1, x_test.shape[1], x_test.shape[0])
            print('x_test shape: ', np.shape(x_test))

            # load model and make forecast for the time serie
            forecaster = models.load_model(''.join([local_script_settings['models_path'],
                                                    'model_group_', str(group),
                                                    '_forecast_.h5']),
                                           custom_objects={'modified_mape': modified_mape,
                                                           'customized_loss': customized_loss})
            point_forecast_normalized = forecaster.predict(x_test)
            print('forecast shape: ', np.shape(point_forecast_normalized))
            print('group: ', group, '\ttime_serie: all ones belonging to this group')
            # inverse reshape
            point_forecast_reshaped = point_forecast_normalized.reshape((point_forecast_normalized.shape[2],
                                                                         point_forecast_normalized.shape[1]))

            # inverse transform (first moving_windows denormalizing and then general rescaling)
            time_serie_normalized_window_mean = np.mean(groups_list[group][:, -moving_window_length:], axis=1)
            group_time_serie_window_scaled_sales_mean = [
                mean_scaled_window_time_serie[[time_serie for time_serie in time_series_in_group]]]
            denormalized_array = window_based_denormalizer(point_forecast_reshaped,
                                                           time_serie_normalized_window_mean,
                                                           time_steps_days)
            group_time_serie_unit_sales_mean = []
            for time_serie in time_series_in_group:
                group_time_serie_unit_sales_mean.append(mean_unit_complete_time_serie[time_serie])
            point_forecast = general_mean_rescaler(denormalized_array,
                                                   np.array(group_time_serie_unit_sales_mean), time_steps_days)
            # point_forecast = np.ceil(point_forecast[0, :, :])
            point_forecast = point_forecast.reshape(np.shape(point_forecast)[1], np.shape(point_forecast)[2])
            point_forecast = point_forecast[:, -forecast_horizon_days:]
            forecasts.append(point_forecast)
            # save points forecast
            np.savetxt(''.join([local_script_settings['others_outputs_path'], 'point_forecast_',
                                '_group_', str(group), '_.csv']), point_forecast, fmt='%10.15f',
                       delimiter=',', newline='\n')
            print('point forecasts saved to file')
        print("forecast subprocess ended successfully")
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' correct forecasting process']))
        forecasts = np.array(forecasts)
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
        for group in range(nof_groups):
            time_series_in_group = time_series_group[:, [0]][time_series_group[:, [1]] == group]
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
            # save the evaluation of models
            time_series_error_mse = np.array(time_series_error_mse)
            time_series_error_mod_mape = np.array(time_series_error_mod_mape)
            time_series_error_mape = np.array(time_series_error_mape)
            y_ground_truth_array = np.array(y_ground_truth_array)
            y_pred_array = np.array(y_pred_array)
            np.save(''.join([local_script_settings['models_evaluation_path'], '_y_ground_truth_array_']),
                    y_ground_truth_array)
            np.savetxt(''.join([local_script_settings['models_evaluation_path'], '_y_pred_array_.csv']),
                       y_pred_array, fmt='%10.15f', delimiter=',', newline='\n')
            np.save(''.join([local_script_settings['models_evaluation_path'], '_y_ground_truth_array_']),
                    y_ground_truth_array)
            np.savetxt(''.join([local_script_settings['models_evaluation_path'], '_y_pred_array_.csv']),
                       y_pred_array, fmt='%10.15f', delimiter=',', newline='\n')
            np.save(''.join([local_script_settings['models_evaluation_path'], 'ts_error_MSE_']), time_series_error_mse)
            np.savetxt(''.join([local_script_settings['models_evaluation_path'], 'ts_error_MSE_.csv']),
                       time_series_error_mse, fmt='%10.15f', delimiter=',', newline='\n')
            np.save(''.join([local_script_settings['models_evaluation_path'], 'ts_error_Mod_MAPE_']),
                    time_series_error_mod_mape)
            np.savetxt(''.join([local_script_settings['models_evaluation_path'], 'ts_error_Mod_MAPE_.csv']),
                       time_series_error_mod_mape, fmt='%10.15f', delimiter=',', newline='\n')
            np.save(''.join([local_script_settings['models_evaluation_path'], 'ts_error_MAPE_']),
                    time_series_error_mape)
            np.savetxt(''.join([local_script_settings['models_evaluation_path'], 'ts_error_MAPE_.csv']),
                       time_series_error_mape, fmt='%10.15f', delimiter=',', newline='\n')
            print('models evaluation metrics for each time serie forecast saved to file')
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' successful saved error metrics']))

        # treating time series with mediocre to bad forecasts (high loss) calling the specific submodule
        if local_script_settings['repeat_training_in_block'] == "True" \
                and local_script_settings['first_train_approach'] != 'stochastic_simulation':
            in_block_time_series_forecast = in_block_high_loss_ts_forecast()
            time_series_reviewed = in_block_time_series_forecast.forecast(local_settings=local_script_settings,
                                                                          local_raw_unit_sales=raw_unit_sales,
                                                                          local_mse=time_series_error_mse)
            print('last step -time_serie specific (in-block) forecast- completed, success: ', time_series_reviewed)

    except Exception as e1:
        print('Error in predict module')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' predict module error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
