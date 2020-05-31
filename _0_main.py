# main program

# libraries
try:
    import json
    import logging
    import logging.handlers as handlers
    import os
    import sys
    from tensorflow.keras import backend as kb
    print('-python libraries loaded')
except Exception as ee1:
    print('Error importing python dependencies')
    print(ee1)

# load settings
try:
    with open('./settings.json') as json_file:
        script_settings = json.loads(json_file.read())
        json_file.close()
    if script_settings['metaheuristic_optimization'] == "True":
        with open(''.join([script_settings['metaheuristics_path'], 'organic_settings.json'])) as json_file:
            script_settings = json.loads(json_file.read())
            json_file.close()
    print('-settings loaded')
except Exception as ee1:
    print('Error reading settings')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)
print('-logging setup completed')

# register current settings in log
logging.info("\nStarting main program..\ncurrent settings:%s", ''.join(['\n', str(script_settings).replace(',', '\n')]))
print('-current settings registered in log')

# custom libraries
try:
    sys.path.insert(0, script_settings['root_path'])
    import _1_prepare_data
    import _2_train
    import _3_predict
    print('-principal modules imported')
    sys.path.insert(1, script_settings['custom_library_path'])
    # here import custom modules
    # from dummy_module import Foo
    print('-custom modules imported')
except Exception as ee1:
    print('Error importing custom modules')
    print(ee1)
    logger.error(str(ee1), exc_info=True)


# main code
if __name__ == '__main__':
    for cycle in range(script_settings['optimization_iterations']):
        preparing, training, predicting = [False] * 3
        try:
            preparing = _1_prepare_data.prepare()
            kb.clear_session()
            training = _2_train.train()
            predicting = _3_predict.predict()
            print('\n~~end of process~~')
        except Exception as ee1:
            print('Error in the main program')
            print(ee1)
            logger.error(str(ee1), exc_info=True)
        print('subprocess', "success?")
        print('preparing: ', preparing)
        print('training: ', training)
        print('predicting: ', predicting)
