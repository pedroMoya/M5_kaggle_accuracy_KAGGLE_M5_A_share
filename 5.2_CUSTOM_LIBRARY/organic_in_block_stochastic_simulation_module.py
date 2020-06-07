# stochastic simulation time series module
import os
import logging
import logging.handlers as handlers
import json
import itertools as it
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as kb

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
np.random.seed(42)
tf.random.set_seed(42)

# clear session
kb.clear_session()


def random_event_realization(local_time_serie_data, local_days_in_focus_frame,
                             local_forecast_horizon_days, local_nof_features_for_training):
    # computing non_zero_frequencies by time_serie, this brings the probability_of_sale = 1 - zero_frequency
    # ---------------kernel----------------------------------
    x_data = local_time_serie_data[:, -local_days_in_focus_frame:]
    probability_of_sale_list = []
    for time_serie in range(local_nof_features_for_training):
        nof_nonzeros = np.count_nonzero(x_data[time_serie, :])
        probability_of_sale_list.append(nof_nonzeros / local_days_in_focus_frame)
    probability_of_sale_array = np.array(probability_of_sale_list)
    # mean with zero included (test with zero excluded, but obtains poorer results)
    mean_last_days_frame = np.mean(x_data[:, -local_days_in_focus_frame:], axis=1)
    # triggering random event and assign sale or not, if sale then fill with mean, if no maintain with zero
    local_y_pred = np.zeros(shape=(local_nof_features_for_training, local_forecast_horizon_days))
    random_event_array_normal = np.random.rand(local_nof_features_for_training, local_forecast_horizon_days)
    pareto_dist = np.random.pareto(3, (local_nof_features_for_training, local_forecast_horizon_days))
    pareto_dist_normalized = pareto_dist / np.amax(pareto_dist)
    random_event_array = np.divide(np.add(4 * pareto_dist_normalized, 3 * random_event_array_normal), 7.)
    local_zero_loc = []
    for time_serie, day in it.product(range(local_nof_features_for_training), range(local_forecast_horizon_days)):
        if probability_of_sale_array[time_serie] > random_event_array[time_serie, day]:
            local_y_pred[time_serie: time_serie + 1, day] = mean_last_days_frame[time_serie]
        else:
            local_y_pred[time_serie: time_serie + 1, day] = 0.
            local_zero_loc.append([time_serie, day])
    return local_y_pred, local_zero_loc
    # ---------------kernel----------------------------------


class organic_in_block_estochastic_simulation:

    def run_stochastic_simulation(self, local_settings, local_raw_unit_sales, local_mse=None):
        try:
            # local_mse will be used if at first train a neural network -> results stored in local_mse
            print('starting time_series in-block forecast submodule')
            # set training parameters
            with open(''.join([local_settings['hyperparameters_path'],
                               'organic_in_block_time_serie_based_model_hyperparameters.json'])) \
                    as local_r_json_file:
                model_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()

            # obtaining representative samples, and assuming a uniform Normal distribution..and Pareto mixed
            forecast_horizon_days = local_settings['forecast_horizon_days']
            event_iterations = model_hyperparameters['event_iterations']
            days_in_focus_frame = model_hyperparameters['days_in_focus_frame']
            nof_features_for_training = nof_time_series = local_raw_unit_sales.shape[0]
            x_data = local_raw_unit_sales[:, -days_in_focus_frame:]
            forecasts = np.zeros(shape=(nof_time_series * 2, forecast_horizon_days))
            # obtaining representative samples, and assuming a uniform Normal distribution..
            mean_stochastic_simulations = []
            median_stochastic_simulations = []
            standard_deviation_stochastic_simulations = []
            zero_loc = []
            # ---------------kernel----------------------------------
            for event in range(event_iterations):
                y_pred, zero_loc = random_event_realization(x_data, days_in_focus_frame,
                                                            forecast_horizon_days, nof_features_for_training)
                standard_deviation_stochastic_simulations.append(np.std(y_pred, axis=1))
                mean_stochastic_simulations.append(np.mean(y_pred, axis=1))
                median_stochastic_simulations.append(np.median(y_pred, axis=1))
            # this statistical values brings confidence interval for the "uncertainty" branch of this competition
            standard_deviation_stochastic_simulations = np.array(standard_deviation_stochastic_simulations)
            mean_stochastic_simulations = np.array(mean_stochastic_simulations)
            median_stochastic_simulations = np.array(median_stochastic_simulations)
            mean_stochastic_simulations = np.mean(mean_stochastic_simulations, axis=0)
            median_stochastic_simulations = np.mean(median_stochastic_simulations, axis=0)
            y_pred_launched = np.divide(np.add(mean_stochastic_simulations, median_stochastic_simulations), 2.)
            standard_deviation_stochastic_simulations = np.mean(standard_deviation_stochastic_simulations, axis=0)
            y_pred = []
            for time_serie in range(nof_features_for_training):
                mu, sigma = median_stochastic_simulations[time_serie], \
                            standard_deviation_stochastic_simulations[time_serie]
                y_pred.append(np.random.normal(mu, sigma, forecast_horizon_days).clip(0))
            y_pred = np.array(y_pred)
            forecasts[:nof_features_for_training, :] = y_pred
            if local_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                forecasts[:nof_features_for_training, :] = local_raw_unit_sales[:, -forecast_horizon_days:]
                forecasts[nof_features_for_training:, :] = y_pred
            # ---------------kernel----------------------------------
            print('StochasticModel computed and simulation done\n')

            # saving submission as organic_submission.csv
            # using template (sample_submission)
            forecast_data_frame = np.genfromtxt(''.join([local_settings['raw_data_path'], 'sample_submission.csv']),
                                                delimiter=',', dtype=None, encoding=None)
            forecast_data_frame[1:, 1:] = forecasts
            pd.DataFrame(forecasts).to_csv(''.join([local_settings['submission_path'],
                                                    'stochastic_simulation_forecasts.csv']),
                                           index=False, header=None)
            zero_loc = np.array(zero_loc)
            np.save(''.join([local_settings['train_data_path'],
                             'stochastic_simulation_forecasts']), forecasts)
            print('stochastic_simulation_submission.csv saved')
            print('organic_in_block_stochastic simulation submodule has finished')
        except Exception as submodule_error:
            print('time_series organic_in_block_stochastic simulation submodule_error: ', submodule_error)
            logger.info('error in organic_in_block_stochastic time_series simulation')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
