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
    from organic_in_block_stochastic_simulation_module import organic_in_block_estochastic_simulation
    from stochastic_model_obtain_results import stochastic_simulation_results_analysis
    from diff_trend_time_serie_module import difference_trends_insight
    from individual_ts_neural_network_training import neural_network_time_serie_schema
    # from mini_module_submission_generator import save_submission
    from save_forecast_and_make_submission import save_forecast_and_submission

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
        if local_script_settings['only_evaluations'] == "True":
            print('follow settings specifications, training is skipped')
            return True
        if local_script_settings['metaheuristic_optimization'] == "True":
            print('changing settings control to metaheuristic optimization')
            with open(''.join(
                    [local_script_settings['metaheuristics_path'], 'organic_settings.json'])) as local_r_json_file:
                local_script_settings = json.loads(local_r_json_file.read())
                local_r_json_file.close()
                metaheuristic_train = tuning_metaheuristic()
                metaheuristic_hyperparameters = metaheuristic_train.stochastic_brain(local_script_settings)
                in_block_time_series_stochastic_simulation = organic_in_block_estochastic_simulation()
                if not metaheuristic_hyperparameters:
                    print('error initializing metaheuristic module')
                    logger.info('error at metaheuristic initialization')
                else:
                    print('metaheuristic module initialized')
                    logger.info('metaheuristic tuning hyperparameters and best_results loaded')

        # opening hyperparameters
        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json'])) \
                as local_r_json_file:
            model_hyperparameters = json.loads(local_r_json_file.read())
            local_r_json_file.close()
        with open(''.join([local_script_settings['hyperparameters_path'],
                           'organic_in_block_time_serie_based_model_hyperparameters.json'])) \
                as local_r_json_file:
            organic_in_block_time_serie_based_model_hyperparameters = \
                json.loads(local_r_json_file.read())
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
    logging.info("\nexecuting train module program..\ncurrent models hyperparameters settings:%s",
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
                return True
        else:
            print('model training start')

        # load raw_data
        raw_data_filename = 'sales_train_evaluation.csv'
        raw_data_sales = pd.read_csv(''.join([local_script_settings['raw_data_path'], raw_data_filename]))
        print('raw sales data accessed')

        # extract data and check  dimensions
        raw_unit_sales = raw_data_sales.iloc[:, 6:].values
        max_selling_time = np.shape(raw_unit_sales)[1]
        local_settings_max_selling_time = local_script_settings['max_selling_time']
        print('max_selling_time(test) inferred by raw data shape:', max_selling_time)
        print('max_selling_time(train) based in settings info:', local_settings_max_selling_time)
        print('It is expected that max_selling_time(train) were at least 28 days lesser than max_selling_time(test)')
        if local_settings_max_selling_time + 28 <= max_selling_time:
            print('and this condition is correctly met')
            raw_unit_sales_ground_truth = raw_unit_sales
            print('length raw data ground truth:', raw_unit_sales_ground_truth.shape[1])
            raw_unit_sales = raw_unit_sales[:, :local_settings_max_selling_time]
            print('length raw data for training:', raw_unit_sales.shape[1])
        elif max_selling_time != local_settings_max_selling_time:
            print("settings doesn't match data dimensions, it must be rechecked before continue(_train_module)")
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
                print('settings indicate that this is the last stage!')
                print('caution: take in consideration that evaluations in this point are not useful, '
                      'because will be made using the last data (the same used in training)')

        # checking correct order in run models
        if local_script_settings['first_train_approach'] == 'stochastic_simulation':
            print('the order in model execution will be: first stochastic_simulation, '
                  'second diff_trends stochastic model, and third neural_network')
        else:
            print('first_train_approach parameter in settings not defined or unknown')
            return False

        # _______________________FIRST_MODEL_______________
        # training in-block stochastic simulation submodule
        print('assuming first_train_approach as stochastic simulation')
        stochastic_simulation = organic_in_block_estochastic_simulation()
        time_series_ss_review, first_model_forecasts = stochastic_simulation.run_stochastic_simulation(
            local_settings=local_script_settings, local_raw_unit_sales=raw_unit_sales)
        print('in-block stochastic simulation completed, success --> ', time_series_ss_review)
        # saving first model forecast and submission bases only in this second model
        store_and_submit_first_model_forecast = save_forecast_and_submission()
        first_model_save_review = \
            store_and_submit_first_model_forecast.store_and_submit('first_model_forecast_data', local_script_settings,
                                                                   first_model_forecasts)
        if first_model_save_review:
            print('first_model forecast data and submission done')
        else:
            print('error at storing first model forecast data or submission')
        # first model results, necessary here because time_series_not_improved is input to second model
        first_model_results = stochastic_simulation_results_analysis()
        time_series_not_improved = first_model_results.evaluate_stochastic_simulation(
            local_script_settings, organic_in_block_time_serie_based_model_hyperparameters, raw_unit_sales,
            raw_unit_sales_ground_truth, 'stochastic_simulation')

        # _______________________SECOND_MODEL_____________________________
        # applying second previous diff-oriented stochastic simulation to high_loss time_series
        # simply --->  "the diffs of immediately previous forecasts improve the last"
        # formula    pred_improved = pred_current + real_previous - pred_previous
        print('\nsecond model (previous difference-oriented trends stochastic simulation)')
        forecast_horizon_days = local_script_settings['forecast_horizon_days']
        time_series_previous_diff_review, previous_pred = stochastic_simulation.run_stochastic_simulation(
            local_settings=local_script_settings, local_raw_unit_sales=raw_unit_sales[:, :-forecast_horizon_days])
        if time_series_previous_diff_review:
            print('previous diff stochastic model training done correctly')
            print(previous_pred)
        else:
            print('error in previous diff stochastic model training')
        ground_truth = raw_unit_sales_ground_truth[:, -2 * forecast_horizon_days: -forecast_horizon_days]
        diff_previous = np.add(ground_truth, -previous_pred[: ground_truth.shape[0], : ground_truth.shape[1]])
        print(diff_previous)
        second_model_forecasts = np.add(diff_previous, first_model_forecasts).clip(0)
        print(second_model_forecasts)
        # saving second model forecast and submission bases only in this second model
        store_and_submit_second_model_forecast = save_forecast_and_submission()
        second_model_save_review = \
            store_and_submit_second_model_forecast.store_and_submit('second_model_forecast_data', local_script_settings,
                                                                    second_model_forecasts)
        if second_model_save_review:
            print('second_model forecast data and submission done')
        else:
            print('error at storing second model forecast data or submission')

        # creating combination of first and second models
        print('nof first model not improved time_series -->', len(time_series_not_improved))
        first_two_models_forecasts_consolidate = first_model_forecasts
        # first_two_models_forecasts_consolidate[time_series_not_improved, :] = \
        #     second_model_forecasts[time_series_not_improved, :]

        first_two_models_forecasts_consolidate[: second_model_forecasts.shape[0], : second_model_forecasts.shape[1]] \
            = np.add(second_model_forecasts, first_two_models_forecasts_consolidate[
                                            : second_model_forecasts.shape[0], : second_model_forecasts.shape[1]])
        first_two_models_forecasts_consolidate = np.divide(first_two_models_forecasts_consolidate, 2)

        # saving second model forecast and submission based only in this second model
        store_and_submit_combination_model_forecast = save_forecast_and_submission()
        combination_model_save_review = \
            store_and_submit_combination_model_forecast.store_and_submit('combination_first_two_model_forecast_data',
                                                                         local_script_settings,
                                                                         first_two_models_forecasts_consolidate)
        if combination_model_save_review:
            print('second_model forecast data and submission done')
        else:
            print('error at storing second model forecast data or submission')

        # second model results, necessary here because time_series_not_improved is input to third model
        first_model_not_improved_ts = len(time_series_not_improved)
        second_model_results = stochastic_simulation_results_analysis()
        time_series_not_improved = second_model_results.evaluate_stochastic_simulation(
            local_script_settings, organic_in_block_time_serie_based_model_hyperparameters,
            raw_unit_sales[:-2 * forecast_horizon_days],
            raw_unit_sales_ground_truth[:-forecast_horizon_days], 'diff_trends_based_stochastic_model')

        # results of COMBINATION of first and second models
        second_model_not_improved_ts = len(time_series_not_improved)
        combination_model_results = stochastic_simulation_results_analysis()
        time_series_not_improved = combination_model_results.evaluate_stochastic_simulation(
            local_script_settings, organic_in_block_time_serie_based_model_hyperparameters, raw_unit_sales,
            raw_unit_sales_ground_truth, 'combination_stochastic_model')

        # results in terms of time_series not_improved
        print('first model nof time_series not improved:', first_model_not_improved_ts)
        print('second model nof time_series not improved:', second_model_not_improved_ts)
        if first_model_not_improved_ts > second_model_not_improved_ts:
            print('applying second model(alone), the results improve in ',
                  first_model_not_improved_ts - second_model_not_improved_ts, ' time_series')
        else:
            print('it is not observed an improvement applying the second model')
        combination_model_not_improved_ts = len(time_series_not_improved)
        if first_model_not_improved_ts < second_model_not_improved_ts:
            best_alone_model_not_improved_ts = first_model_not_improved_ts
        else:
            best_alone_model_not_improved_ts = second_model_not_improved_ts
        print('best (first or second) model nof time_series not improved:', best_alone_model_not_improved_ts)
        print('combination model nof time_series not improved:', combination_model_not_improved_ts)
        if best_alone_model_not_improved_ts > combination_model_not_improved_ts:
            print('applying combination model, the results improve in ',
                  best_alone_model_not_improved_ts - combination_model_not_improved_ts, ' time_series')

        else:
            print('it is not observed an improvement applying the combination model')

        while True:
            a = 0

        # _______________________THIRD_MODEL_____________________________
        # training individual_time_serie with specific time_serie LSTM-ANN
        print('\nrunning third model (neural_network)')
        neural_network_ts_schema_training = neural_network_time_serie_schema()
        training_nn_review = neural_network_ts_schema_training.train(local_script_settings,
                                                                     raw_unit_sales, model_hyperparameters,
                                                                     time_series_not_improved,
                                                                     raw_unit_sales_ground_truth)
        print('neural_network model trained, with success -->', training_nn_review)

        # closing train module
        print('full training module ended')
        if training_nn_review and time_series_ss_review:
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' correct model training, correct saving of model and weights']))
            local_script_settings['training_done'] = "True"
        else:
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
    except Exception as e1:
        print('Error training model')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' model training error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
