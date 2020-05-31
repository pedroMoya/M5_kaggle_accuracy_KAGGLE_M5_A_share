# Metaheuristic module
import os
import logging
import logging.handlers as handlers
import json
import numpy as np

# open local settings
with open('./settings.json') as local_json_file:
    local_script_settings = json.loads(local_json_file.read())
    local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# set random seed for reproducibility --> done in _2_train.py module
# np.random.seed(3)


class tuning_metaheuristic:

    @staticmethod
    def load_hyperparameters_options(local_metaheuristic_options):
        try:
            learning_rate_list = local_metaheuristic_options['learning_rate']
            optimizer_list = local_metaheuristic_options['optimizer']
            batch_size_list = local_metaheuristic_options['batch_size']
            epochs_list = local_metaheuristic_options['epochs']
            stride_window_walk_list = local_metaheuristic_options['stride_window_walk']
            days_in_focus_frame_list = local_metaheuristic_options['days_in_focus_frame']
            activation_1_list = local_metaheuristic_options['activation_1']
            activation_2_list = local_metaheuristic_options['activation_2']
            activation_3_list = local_metaheuristic_options['activation_3']
            activation_4_list = local_metaheuristic_options['activation_4']
            unit_layer_1_list = local_metaheuristic_options['units_layer_1']
            unit_layer_2_list = local_metaheuristic_options['units_layer_2']
            unit_layer_3_list = local_metaheuristic_options['units_layer_3']
            unit_layer_4_list = local_metaheuristic_options['units_layer_4']
            loss_function_list = local_metaheuristic_options['loss_1']
            dropout_layer_1_list = local_metaheuristic_options['dropout_layer_1']
            dropout_layer_2_list = local_metaheuristic_options['dropout_layer_2']
            dropout_layer_3_list = local_metaheuristic_options['dropout_layer_3']
            dropout_layer_4_list = local_metaheuristic_options['dropout_layer_4']
            model_type_list = local_metaheuristic_options['model_type']
            list_of_options =\
                [
                    learning_rate_list,
                    optimizer_list,
                    batch_size_list,
                    epochs_list,
                    stride_window_walk_list,
                    days_in_focus_frame_list,
                    activation_1_list,
                    activation_2_list,
                    activation_3_list,
                    activation_4_list,
                    unit_layer_1_list,
                    unit_layer_2_list,
                    unit_layer_3_list,
                    unit_layer_4_list,
                    loss_function_list,
                    dropout_layer_1_list,
                    dropout_layer_2_list,
                    dropout_layer_3_list,
                    dropout_layer_4_list,
                    model_type_list,
                ]
        except Exception as submodule_error:
            print('metaheuristic load_hyperparameters submodule_error: ', submodule_error)
            return False
        return list_of_options

    def stochastic_brain(self, local_settings):
        try:
            with open(''.join(
                    [local_settings['metaheuristics_path'], 'metaheuristic_options.json'])) as local_r_json_file:
                metaheuristic_options = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            list_of_options = self.load_hyperparameters_options(metaheuristic_options)
            hyperparameters_option_length = [len(hyperparameter_option) for hyperparameter_option in list_of_options]
            with open(''.join(
                    [local_settings['metaheuristics_path'], 'metaheuristic_results.json'])) as local_r_json_file:
                metaheuristic_results = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            last_best_result_mse = metaheuristic_results['last_best_result_mse']
            last_best_setup = metaheuristic_results['last_best_setup']
            random_generator_list = [np.random.randint(0, hyperparameters_option_length[hyperparameter])
                                     for hyperparameter in range(len(list_of_options))]
            hyperparameters = [list_of_options[hyperparameter_index][random_generator_list[hyperparameter_index]]
                               for hyperparameter_index in range(len(list_of_options))]
            if last_best_result_mse == "None":
                print('no old data, initializing hyperparameters..')
                print('hyperparameters initialised')
            else:
                # 0 no make change, 1 make change
                random_generator_changes = [np.random.randint(0, 2)] * len(hyperparameters)
                for hyperparameter_index in range(len(hyperparameters)):
                    if random_generator_changes[hyperparameter_index] == 0:
                        hyperparameters[hyperparameter_index] = last_best_setup[hyperparameter_index]
            print('hyperparameters:')
            print(hyperparameters)
        except Exception as submodule_error:
            print('metaheuristic stochastic submodule_error: ', submodule_error)
            logger.info('error in random generator change (metaheuristic_module)')
            return False
        with open(''.join([local_settings['metaheuristics_path'],
                           'organic_model_hyperparameters.json']), 'r', encoding='utf-8') as local_r_json_file:
            organic_model_hyperparameters = json.loads(local_r_json_file.read())
            local_r_json_file.close()
        for hyperparameter_name, new_value in zip(metaheuristic_options, hyperparameters):
            print(hyperparameter_name, new_value)
            organic_model_hyperparameters[hyperparameter_name] = new_value
        with open(''.join([local_settings['metaheuristics_path'],
                           'organic_model_hyperparameters.json']), 'w', encoding='utf-8') as local_wr_json_file:
            json.dump(organic_model_hyperparameters, local_wr_json_file, ensure_ascii=False, indent=2)
            local_wr_json_file.close()
        return True

    def evaluation_brain(self, local_result_mse, local_settings):
        try:
            with open(''.join(
                    [local_settings['metaheuristics_path'], 'metaheuristic_options.json'])) as local_r_json_file:
                metaheuristic_options = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            with open(''.join(
                    [local_settings['metaheuristics_path'], 'metaheuristic_results.json'])) as local_r_json_file:
                metaheuristic_results = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            last_best_result_mse = metaheuristic_results['last_best_result_mse']
            print('last best mse: ', last_best_result_mse)
            print('current mse: ', local_result_mse)
            better = False
            if last_best_result_mse == "None":
                print('no previous last best result stored')
            elif last_best_result_mse < local_result_mse:
                return True, better
            else:
                better = True
            # only execute this code if there is not previous result or if the result is better
            metaheuristic_results['last_best_result_mse'] = local_result_mse
            with open(''.join([local_settings['metaheuristics_path'],
                               'organic_model_hyperparameters.json']), 'r', encoding='utf-8') as local_r_json_file:
                organic_model_hyperparameters = json.loads(local_r_json_file.read())
                local_r_json_file.close()
            hyperparameters = []
            [hyperparameters.append(organic_model_hyperparameters[hyperparameter_name])
             for hyperparameter_name in metaheuristic_options]
            print(hyperparameters)
            metaheuristic_results['last_best_setup'] = hyperparameters
            print(metaheuristic_results)
            with open(''.join([local_settings['metaheuristics_path'],
                               'metaheuristic_results.json']), 'w', encoding='utf-8') as local_wr_json_file:
                json.dump(metaheuristic_results, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
            print('metaheuristic submodule completed without errors')
            return True, better
        except Exception as module_error:
            print('model_evaluation metaheuristic submodule_error: ', module_error)
            logger.info('error in model_evaluation metaheuristic_submodule')
            return False, None
