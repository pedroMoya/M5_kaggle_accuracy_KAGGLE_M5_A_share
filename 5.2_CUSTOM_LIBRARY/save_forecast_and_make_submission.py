# saving forecasts and making forecast
import os
import sys
import logging
import logging.handlers as handlers
import json
import pandas as pd
import numpy as np

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

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from mini_module_submission_generator import save_submission


class save_forecast_and_submission:

    def store_and_submit(self, local_name, local_settings, local_y_pred):
        try:
            nof_time_series = local_settings['number_of_time_series']
            nof_days = local_y_pred.shape[1]
            local_forecast_horizon_days = local_settings['forecast_horizon_days']
            local_forecasts = np.zeros(shape=(nof_time_series * 2, nof_days), dtype=np.dtype('float32'))
            local_forecasts[:nof_time_series, :] = local_y_pred

            # dealing with Validation stage or Evaluation stage
            if local_settings['competition_stage'] == 'submitting_after_June_1th_using_1941days':
                local_raw_data_filename = 'sales_train_evaluation.csv'
                local_raw_data_sales = pd.read_csv(''.join([local_settings['raw_data_path'], local_raw_data_filename]))
                local_raw_unit_sales = local_raw_data_sales.iloc[:, 6:].values
                local_forecasts[:nof_time_series, :] = local_raw_unit_sales[:, -local_forecast_horizon_days:]
                local_forecasts[nof_time_series:, :] = local_y_pred

            # saving forecast data
            np.save(''.join([local_settings['train_data_path'], local_name]), local_forecasts)

            # saving submission as (local_name).csv
            save_submission_stochastic_simulation = save_submission()
            save_submission_review = \
                save_submission_stochastic_simulation.save(''.join([local_name, '_submission.csv']), local_forecasts,
                                                           local_settings)
            if save_submission_review:
                print('submission of model forecasts successfully completed')
            else:
                print('error at saving submission')

        except Exception as submodule_error:
            print('save_forecast and make_submission submodule_error: ', submodule_error)
            logger.info('error in save_forecast and make submission submodule')
            logger.error(str(submodule_error), exc_info=True)
            return False
        return True
