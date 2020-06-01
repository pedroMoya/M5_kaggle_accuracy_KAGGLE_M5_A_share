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
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float32')
    from tensorflow.keras import layers
    from tensorflow.keras.experimental import PeepholeLSTMCell
    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras.layers import RepeatVector
    from tensorflow.keras import backend as kb
    from tensorflow.keras import regularizers
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras import metrics
    from tensorflow.keras import callbacks as cb

    # open local settings
    with open('./settings.json') as local_json_file:
        local_settings = json.loads(local_json_file.read())
        local_json_file.close()
    sys.path.insert(1, local_settings['custom_library_path'])
    from metaheuristic_module import tuning_metaheuristic
    from organic_in_block_high_loss_identified_ts_forecast_module import in_block_high_loss_ts_forecast
except Exception as ee1:
    print('Error importing libraries or opening settings (train module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# keras session, random seed reset/fix, set_epsilon for keras backend
kb.clear_session()
np.random.seed(11)
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


def train():
    # load model hyparameters
    try:
        with open('./settings.json') as local_r_json_file:
            local_script_settings = json.loads(local_r_json_file.read())
            local_r_json_file.close()
        if local_script_settings['metaheuristic_optimization'] == "True":
            print('changing settings control to metaheuristic optimization')
            with open(''.join(
                    [local_script_settings['metaheuristics_path'], 'organic_settings.json'])) as local_r_json_file:
                local_script_settings = json.loads(local_r_json_file.read())
                local_r_json_file.close()
                metaheuristic_train = tuning_metaheuristic()
                metaheuristic_hyperparameters = metaheuristic_train.stochastic_brain(local_script_settings)
                in_block_time_series_forecast = in_block_high_loss_ts_forecast()
                if not metaheuristic_hyperparameters:
                    print('error initializing metaheuristic module')
                    logger.info('error at metaheuristic initialization')
                else:
                    print('metaheuristic module initialized')
                    logger.info('metaheuristic tuning hyperparameters and best_results loaded')

        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json'])) \
                as local_r_json_file:
            model_hyperparameters = json.loads(local_r_json_file.read())
            local_r_json_file.close()
        if local_script_settings['data_cleaning_done'] == 'True' and \
                model_hyperparameters['time_steps_days'] != local_script_settings['time_steps_days']:
            model_hyperparameters['time_steps_days'] = local_script_settings['time_steps_days']
            print('during load of train module, a recent change in time_steps was detected,')
            print('in order to maintain consistency cleaning of data and training of model will be repeated')
            local_script_settings['data_cleaning_done'] = 'False'
            local_script_settings['training_done'] = 'False'
            with open('./settings.json', 'w', encoding='utf-8') as local_w_json_file:
                json.dump(local_script_settings, local_w_json_file, ensure_ascii=False, indent=2)
                local_w_json_file.close()
        else:
            print('check of metadata consistency passed without found problems')
            print('but, please verify that data prepare was done in fact with the last corrections in time_steps. '
                  'Consider repeat data cleaning and model training if it is necessary')
        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json']),
                  'w', encoding='utf-8') as local_w_json_file:
            json.dump(model_hyperparameters, local_w_json_file, ensure_ascii=False, indent=2)
            local_w_json_file.close()
            print('time_step_days conciliated:', model_hyperparameters['time_steps_days'], ' (_train_ module check)')

    except Exception as e1:
        print('Error loading LTSM model hyperparameters (train module)')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' error at loading model (LTSM) hyperparameters']))
        logger.error(str(e1), exc_info=True)

    # register model hyperparameters settings in log
    logging.info("\nstarting main program..\ncurrent models hyperparameters settings:%s",
                 ''.join(['\n', str(model_hyperparameters).replace(',', '\n')]))
    print('-current models hyperparameters registered in log')
    try:
        print('\n~train_model module~')
        # check settings for previous training and then repeat or not this phase
        if local_script_settings['training_done'] == "True":
            print('training of neural_network previously done')
            if local_script_settings['repeat_training'] == "True":
                print('repeating training')
            else:
                print("settings indicates don't repeat training")
            if local_script_settings['first_train_approach'] == 'neural_network':
                return True
        else:
            print('model training start')

        # load raw_data
        raw_data_sales = pd.read_csv(''.join([local_script_settings['raw_data_path'],
                                              'sales_train_validation.csv']))
        print('raw sales data accessed')

        # extract data and check  dimensions
        raw_unit_sales = raw_data_sales.iloc[:, 6:].values
        if local_script_settings['repeat_training_in_block'] == "True":
            # extract data and check  dimensions
            raw_unit_sales = raw_data_sales.iloc[:, 6:].values
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
                print(''.join(['\x1b[0;2;41m', 'Warning', '\x1b[0m']))
                print('please check: forecast horizon is included within training data')
            print('raw data input collected and check of data dimensions passed (train_module)')

        # load clean data (divided in groups) and groups data
        nof_groups = local_script_settings['number_of_groups']
        window_normalized_scaled_unit_sales = np.load(''.join([local_script_settings['train_data_path'],
                                                               'x_train_source.npy']))
        if nof_groups == 1 or local_settings['first_train_approach'] == "stochastic_simulation":
            scaled_unit_sales_g1 = window_normalized_scaled_unit_sales
            scaled_unit_sales_g2 = np.array([])
            scaled_unit_sales_g3 = np.array([])
            nof_time_series_list = [np.shape(window_normalized_scaled_unit_sales)[0]]
            max_selling_time_list = [local_script_settings['max_selling_time']]
            groups_list = [window_normalized_scaled_unit_sales]
            nof_groups = 1
        else:
            scaled_unit_sales_g1 = np.load(''.join([local_script_settings['train_data_path'], 'group1.npy']))
            scaled_unit_sales_g2 = np.load(''.join([local_script_settings['train_data_path'], 'group2.npy']))
            scaled_unit_sales_g3 = np.load(''.join([local_script_settings['train_data_path'], 'group3.npy']))
            groups_list = [scaled_unit_sales_g1, scaled_unit_sales_g2, scaled_unit_sales_g3]
            # conform data structures and (inside loop) train model for each time serie
            max_selling_time_g1 = np.shape(scaled_unit_sales_g1)[1]
            max_selling_time_g2 = np.shape(scaled_unit_sales_g2)[1]
            max_selling_time_g3 = np.shape(scaled_unit_sales_g3)[1]
            max_selling_time_list = [max_selling_time_g1, max_selling_time_g2, max_selling_time_g3]
            # store the number of time_series in each group,
            # remember that here each time_serie is both a forecast_variable and a feature
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
                number_of_time_series_g1 = forecast_settings['number_of_time_series_g1']
                number_of_time_series_g2 = forecast_settings['number_of_time_series_g2']
                number_of_time_series_g3 = forecast_settings['number_of_time_series_g3']
            nof_time_series_list = [number_of_time_series_g1, number_of_time_series_g2, number_of_time_series_g3]
        if local_script_settings['first_train_approach'] == 'stochastic_simulation':
            print('assuming first_train_approach as stochastic simulation')
            in_block_time_series_forecast = in_block_high_loss_ts_forecast()
            time_series_reviewed = in_block_time_series_forecast.forecast(local_settings=local_script_settings,
                                                                          local_raw_unit_sales=raw_unit_sales)
            time_series_not_improved = np.load(''.join([local_script_settings['models_evaluation_path'],
                                                        'time_series_not_improved.npy']), allow_pickle=True)
            groups_list = [window_normalized_scaled_unit_sales[time_series_not_improved, :]]
            print('-time_serie specific (in-block) forecast- completed, success: ', time_series_reviewed)
        elif local_script_settings['first_train_approach'] != 'neural_network':
            print('first_train_approach parameter in settings not defined or unknown')
            return False
        else:
            print('assuming first_train_approach as neural_network')
        if local_script_settings['repeat_training'] == "False":
            print("settings indicates don't repeat neural_network training")
            print('train module has finished')
            return True

        # set training parameters
        time_steps_days = int(local_script_settings['time_steps_days'])
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

        # create and compile LTSM model
        forecaster_models_list = []
        try:
            for group in range(nof_groups):
                if model_hyperparameters['features'] == 'aggregated_by_group':
                    nof_features_in_group = nof_time_series_list[group]
                else:
                    logging.info("\nnumber of features in settings don't match valid expected values")
                    print("-number of features doesn't match valid expected values")
                    return False
                if local_settings['first_train_approach'] == 'stochastic_simulation':
                    nof_features_in_group = np.shape(groups_list[group])[0]
                forecaster = tf.keras.Sequential()
                if model_hyperparameters['model_type'] == 'Pure_ANN':
                    # model Bidirectional_PeepHole_LSTM
                    print('model: Pure_ANN')
                    # first layer
                    if model_hyperparameters['units_layer_1'] > 0:
                        forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_1'],
                                                    activation=model_hyperparameters['activation_1'],
                                                    input_shape=(local_script_settings['time_steps_days'],
                                                                 nof_features_in_group),
                                                    activity_regularizer=activation_regularizer))
                        forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_1'])))
                    # second layer
                    if model_hyperparameters['units_layer_2'] > 0:
                        forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_2'],
                                                    activation=model_hyperparameters['activation_2'],
                                                    activity_regularizer=activation_regularizer))
                        forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_2'])))
                    # third layer
                    if model_hyperparameters['units_layer_3'] > 0:
                        forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_3'],
                                                    activation=model_hyperparameters['activation_3'],
                                                    activity_regularizer=activation_regularizer))
                        forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_3'])))
                    #  fourth layer
                    if model_hyperparameters['units_layer_4'] > 0:
                        forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_4'],
                                                    activation=model_hyperparameters['activation_4'],
                                                    activity_regularizer=activation_regularizer))
                        forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_4'])))
                    # final layer
                    forecaster.add(layers.Dense(units=nof_features_in_group))
                elif model_hyperparameters['model_type'] == 'PeepHole_Encode_Decode':
                    # Model PeepHole_encode_decode LSTM model
                    # first layer
                    print('model: PeepHoleLSTM_Encode_Decode')
                    if model_hyperparameters['units_layer_1'] > 0:
                        forecaster.add(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_1'],
                                             activation=model_hyperparameters['activation_1'],
                                             input_shape=(model_hyperparameters['time_steps_days'],
                                                          nof_features_in_group),
                                             dropout=float(model_hyperparameters['dropout_layer_1'])),
                            return_sequences=False))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # second layer
                    if model_hyperparameters['units_layer_2'] > 0:
                        forecaster.add(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_2'],
                                             activation=model_hyperparameters['activation_2'],
                                             dropout=float(model_hyperparameters['dropout_layer_2'])),
                            return_sequences=False))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # third layer
                    if model_hyperparameters['units_layer_3'] > 0:
                        forecaster.add(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_3'],
                                             activation=model_hyperparameters['activation_3'],
                                             dropout=float(model_hyperparameters['dropout_layer_3'])),
                            return_sequences=False))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # fourth layer
                    if model_hyperparameters['units_layer_4'] > 0:
                        forecaster.add(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_4'],
                                             activation=model_hyperparameters['activation_4'],
                                             dropout=float(model_hyperparameters['dropout_layer_4'])),
                            return_sequences=False))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # final layer
                    forecaster.add(TimeDistributed(layers.Dense(units=nof_features_in_group)))
                elif model_hyperparameters['model_type'] == 'Bidirectional_PeepHoleLSTM_Encode_Decode':
                    print('current model: Bidirectional_PeepHoleLSTM_Encode_Decode')
                    # first layer
                    if model_hyperparameters['units_layer_1'] > 0:
                        forecaster.add(layers.Bidirectional(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_1'],
                                             activation=model_hyperparameters['activation_1'],
                                             activity_regularizer=activation_regularizer,
                                             input_shape=(model_hyperparameters['time_steps_days'],
                                                          nof_features_in_group),
                                             dropout=float(model_hyperparameters['dropout_layer_1'])),
                            return_sequences=False)))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # second layer
                    if model_hyperparameters['units_layer_2'] > 0:
                        forecaster.add(layers.Bidirectional(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_2'],
                                             activation=model_hyperparameters['activation_2'],
                                             activity_regularizer=activation_regularizer,
                                             dropout=float(model_hyperparameters['dropout_layer_2'])),
                            greturn_sequences=False)))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # third layer
                    if model_hyperparameters['units_layer_3'] > 0:
                        forecaster.add(layers.Bidirectional(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_3'],
                                             activation=model_hyperparameters['activation_3'],
                                             activity_regularizer=activation_regularizer,
                                             dropout=float(model_hyperparameters['dropout_layer_3'])),
                            return_sequences=False)))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # fourth layer
                    if model_hyperparameters['units_layer_4'] > 0:
                        forecaster.add(layers.Bidirectional(layers.RNN(
                            PeepholeLSTMCell(units=model_hyperparameters['units_layer_4'],
                                             activation=model_hyperparameters['activation_4'],
                                             activity_regularizer=activation_regularizer,
                                             dropout=float(model_hyperparameters['dropout_layer_4'])),
                            return_sequences=False)))
                        forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # final layer
                    forecaster.add(TimeDistributed(layers.Dense(units=nof_features_in_group)))
                elif model_hyperparameters['model_type'] == 'Mix_Bid_PeepHole_LSTM_Dense_ANN':
                    print('current model: Mix_Bid_PeepHole_LSTM_Dense_ANN')
                    # first layer (DENSE)
                    if model_hyperparameters['units_layer_1'] > 0:
                        # strictly dim 1 of input_shape is ['time_steps_days'] (dim 0 is number of batches: None)
                        forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_1'],
                                                    activation=model_hyperparameters['activation_1'],
                                                    input_shape=(local_script_settings['time_steps_days'],
                                                                 nof_features_in_group),
                                                    activity_regularizer=activation_regularizer))
                        forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_1'])))
                    # second layer
                    forecaster.add(layers.Bidirectional(layers.RNN(
                        PeepholeLSTMCell(units=model_hyperparameters['units_layer_2'],
                                         activation=model_hyperparameters['activation_2'],
                                         activity_regularizer=activation_regularizer,
                                         dropout=float(model_hyperparameters['dropout_layer_2'])),
                        return_sequences=False)))
                    forecaster.add(RepeatVector(model_hyperparameters['repeat_vector']))
                    # third layer
                    if model_hyperparameters['units_layer_3'] > 0:
                        forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_3'],
                                                    activation=model_hyperparameters['activation_3'],
                                                    activity_regularizer=activation_regularizer))
                        forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_3'])))
                    # fourth layer (DENSE)
                    if model_hyperparameters['units_layer_4'] > 0:
                        forecaster.add(layers.Dense(units=model_hyperparameters['units_layer_4'],
                                                    activation=model_hyperparameters['activation_4'],
                                                    activity_regularizer=activation_regularizer))
                        forecaster.add(layers.Dropout(rate=float(model_hyperparameters['dropout_layer_4'])))
                    # final layer
                    forecaster.add(layers.Dense(units=nof_features_in_group))
                else:
                    print('model not known: ', model_hyperparameters['model_type'])
                    logger.info('model unknown')
                    return False
                forecaster.compile(optimizer=optimizer_function,
                                   loss=losses_list,
                                   metrics=metrics_list)
                forecaster_models_list.append(forecaster)
                print('group ', group, ' LSTM model initialization and compile done')
                # saving untrained models in JSON format
                forecaster_group_json = forecaster.to_json()
                with open(''.join([local_settings['models_path'], 'forecaster_group', str(group),
                                   '_.json']), 'w') as json_file:
                    json_file.write(forecaster_group_json)
                    json_file.close()
                logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                     ' correct LSTM model creation and compilation']))
        except Exception as e1:
            print('Error creating or compiling LTSM model')
            print(e1)
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' LSTM model creation or compile error']))
            logger.error(str(e1), exc_info=True)

        # preparing training inputs arrays structures, callbacks and training of models
        nof_years = local_script_settings['number_of_years_ceil']
        days_in_focus_frame = model_hyperparameters['days_in_focus_frame']
        window_input_length = local_script_settings['moving_window_input_length']
        window_output_length = local_script_settings['moving_window_output_length']
        moving_window_length = window_input_length + window_output_length
        group_iterator = 0
        for group in groups_list:
            nof_features_in_group = group.shape[0]
            nof_selling_days = group.shape[1]
            print('nof_time_series: ', nof_features_in_group)
            last_learning_day_in_year = np.mod(nof_selling_days, 365)
            max_selling_time = max_selling_time_list[group_iterator]
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
            x_train = []
            if local_settings['train_model_input_data_approach'] == "all":
                [x_train.append(group[:, day - time_steps_days: day - window_output_length])
                 for day in range(time_steps_days, max_selling_time, stride_window_walk)]
            elif local_settings['train_model_input_data_approach'] == "focused":
                [x_train.append(group[:, day: day + time_steps_days])
                 for last_day in day_in_year[:-1]
                 for day in range(last_day + window_output_length,
                                  last_day + window_output_length - days_in_focus_frame, -stride_window_walk)]
                # border condition, take care with last year, working with last data available, yeah really!!
                [x_train.append(np.concatenate(
                    (group[:, day - window_output_length: day],
                     np.zeros(shape=(nof_features_in_group, time_steps_days - window_output_length))),
                    axis=1))
                    for last_day in day_in_year[-1:]
                    for day in range(last_day, last_day - days_in_focus_frame, -stride_window_walk)]
            else:
                logging.info("\ntrain_model_input_data_approach is not defined")
                print('-a problem occurs with the data_approach settings')
                return False, None
            y_train = []
            if local_settings['train_model_input_data_approach'] == "all":
                [y_train.append(group[:, day - time_steps_days: day])
                 for day in range(time_steps_days, max_selling_time, stride_window_walk)]
            elif local_settings['train_model_input_data_approach'] == "focused":
                [y_train.append(group[:, day: day + time_steps_days])
                 for last_day in day_in_year[:-1]
                 for day in range(last_day + window_output_length,
                                  last_day + window_output_length - days_in_focus_frame, -stride_window_walk)]
                # border condition, take care with last year, working with last data available, yeah really!!
                [y_train.append(np.concatenate(
                    (group[:, day - window_output_length: day],
                     np.zeros(shape=(nof_features_in_group, time_steps_days - window_output_length))),
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

            # extreme ways.. better results!,
            # zeros control by creating more! because this number is a placeholder for different issues, let's think..
            # that's the key: one of various possible meaning of zero: (1) zero = (stock > 0 and no sales)
            # (2) other meaning = no current stock, (3) other = item no yet in inventory (but in the future yes)
            # (4) other meaning of zero = item never in inventory, (5) other meaning = lost of data for this day
            # an so on..(6) = item sold but latter in the same day returned and refunded
            # eliminate intermediate positive values in x_train and y_train, replacing with max amplified by factor
            if local_script_settings['amplification'] == "True":
                factor = local_script_settings['amplification_factor']  # factor tuning was done previously
                for time_serie_iterator in range(np.shape(x_train)[1]):
                    max_time_serie = np.amax(x_train[:, time_serie_iterator, :])
                    x_train[:, time_serie_iterator, :][x_train[:, time_serie_iterator, :] > 0] = max_time_serie * factor
                    max_time_serie = np.amax(y_train[:, time_serie_iterator, :])
                    y_train[:, time_serie_iterator, :][y_train[:, time_serie_iterator, :] > 0] = max_time_serie * factor
                # with amin the results worse, so don't use, conserve the code just in case
                # min_time_serie = np.amin(x_train[:, time_serie_iterator, :])
                # x_train[:, time_serie_iterator, :][x_train[:, time_serie_iterator, :] < 0] = min_time_serie * factor
                # min_time_serie = np.amin(y_train[:, time_serie_iterator, :])
                # y_train[:, time_serie_iterator, :][y_train[:, time_serie_iterator, :] < 0] = min_time_serie * factor

            # define callbacks, checkpoints namepaths
            model_weights = ''.join([local_script_settings['checkpoints_path'], 'group_', str(group_iterator),
                                     model_hyperparameters['current_model_name'], "_loss_-{loss:.4f}-.hdf5"])
            callback1 = cb.EarlyStopping(monitor='loss', patience=model_hyperparameters['early_stopping_patience'])
            callback2 = cb.ModelCheckpoint(model_weights, monitor='loss', verbose=1,
                                           save_best_only=True, mode='min')
            callbacks = [callback1, callback2]

            x_train = x_train.reshape((np.shape(x_train)[0], np.shape(x_train)[2], np.shape(x_train)[1]))
            y_train = y_train.reshape((np.shape(y_train)[0], np.shape(y_train)[2], np.shape(y_train)[1]))
            print('input_x_shape: ', np.shape(x_train))
            print('input_y_shape: ', np.shape(y_train))

            # train for each group
            forecaster = forecaster_models_list[group_iterator]
            forecaster.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, workers=workers,
                           callbacks=callbacks, shuffle=False)
            forecaster.summary()  # print summary (informative, but if says "shape = multiple"; probably useless!!)

            # store time_serie trained and group
            # model_by_group_trained.append([time_serie, group_iterator])

            # save model for this time serie
            forecaster.save(''.join([local_script_settings['models_path'],
                                     'model_group_', str(group_iterator), '_forecast_.h5']))
            group_iterator += 1

            # saving trained models in JSON format
            forecaster_group_json = forecaster.to_json()
            with open(''.join([local_settings['models_path'], 'trained_forecaster_group', str(group_iterator),
                               '_.json']), 'w') as json_file:
                json_file.write(forecaster_group_json)
                json_file.close()
        print('models successfully trained and saved')

        # save settings changes, if is configured to
        if local_script_settings['automatic_time_series_number'] == "True":
            with open(''.join([local_script_settings['test_data_path'], 'forecast_settings.json']),
                      'r', encoding='utf-8') as local_r_json_file:
                forecast_settings = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            if nof_groups == 1:
                forecast_settings['number_of_forecasts_g1'] = nof_time_series_list[0]
                forecast_settings['number_of_forecasts_g2'] = 0
                forecast_settings['number_of_forecasts_g3'] = 0
            else:
                forecast_settings['number_of_forecasts_g1'] = int(np.shape(scaled_unit_sales_g1)[0])
                forecast_settings['number_of_forecasts_g2'] = int(np.shape(scaled_unit_sales_g2)[0])
                forecast_settings['number_of_forecasts_g3'] = int(np.shape(scaled_unit_sales_g3)[0])
            with open(''.join([local_script_settings['test_data_path'], 'forecast_settings.json']),
                      'w', encoding='utf-8') as local_wr_json_file:
                json.dump(forecast_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_json_file.close()
                print('number of time series saved')
                logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                     ' time series number (by group) saved to forecast settings']))
        print("training ended successfully")
        print("models and weights saved")
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' correct model training, correct saving of model and weights']))
        local_script_settings['training_done'] = "True"
        if local_script_settings['metaheuristic_optimization'] == "False":
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        elif local_script_settings['metaheuristic_optimization'] == "True":
            with open(''.join([local_script_settings['metaheuristics_path'],
                               'organic_settings.json']), 'w', encoding='utf-8') as local_wr_json_file:
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' settings modified and saved']))
        print('raw datasets cleaned, settings saved..')
    except Exception as e1:
        print('Error training model')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' model training error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
