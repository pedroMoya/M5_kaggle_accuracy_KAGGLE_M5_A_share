# evaluate -in a one time_serie by one time_serie only- using mse metric

# importing python libraries and opening settings
try:
    import os
    import datetime
    import logging
    import logging.handlers as handlers
    import json
    from sklearn.metrics import mean_squared_error
    with open('./settings.json') as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()
except Exception as ee1:
    print('Error importing libraries or opening settings (mini_evaluator_ts sub_module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# classes definitions


class mini_evaluator_submodule:

    def evaluate_ts_forecast(self, local_true, local_pred):
        try:
            local_error_metric_mse = mean_squared_error(local_true, local_pred)
        except Exception as e1:
            print('Error in mini_evaluator time_series sub_module')
            print(e1)
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' mini_evaluator_ts sub_module error']))
            logger.error(str(e1), exc_info=True)
            return False
        return local_error_metric_mse
