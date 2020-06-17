# preparing data (cleaning raw data, aggregating and saving to file)

# importing python libraries and opening settings
try:
    import os
    import logging
    import logging.handlers as handlers
    import json
    import datetime
    import numpy as np
    import pandas as pd
    import itertools as it

    # open local settings and change local_scrip_settings if metaheuristic equals True
    with open('./settings.json') as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()
    if local_script_settings['metaheuristic_optimization'] == "True":
        with open(''.join([local_script_settings['metaheuristics_path'],
                           'organic_settings.json'])) as local_json_file:
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
    logger.info('_prepare_data module start')

    # Random seed fixed
    np.random.seed(1)

    # check time_steps
    time_steps_days = local_script_settings['time_steps_days']
    with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json'])) \
            as local_json_file:
        model_hyperparameters = json.loads(local_json_file.read())
        local_json_file.close()
        if model_hyperparameters['time_steps_days'] != time_steps_days:
            model_hyperparameters['time_steps_days'] = time_steps_days
            print('during load of prepare module, a recent change in time_steps was detected,')
            print('in order to maintain and ensure data consistency, '
                  'cleaning and training of model will be repeated')
            local_script_settings['data_cleaning_done'] = 'False'
            local_script_settings['training_done'] = 'False'
            with open('./settings.json', 'w', encoding='utf-8') as local_w_json_file:
                json.dump(local_script_settings, local_w_json_file, ensure_ascii=False, indent=2)
                local_w_json_file.close()
            logger.info('time_steps reconciled ')
        else:
            print('verify /100% be sure\\ that data prepare was done in fact with the last corrections in time_steps,'
                  'if not, better take in mind to repeat data cleaning and model training')
        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json']),
                  'w', encoding='utf-8') as local_w_json_file:
            json.dump(model_hyperparameters, local_w_json_file, ensure_ascii=False, indent=2)
            local_w_json_file.close()
            print('time_step_days conciliated:', model_hyperparameters['time_steps_days'], ' (_prepare_module check)')
except Exception as ee1:
    print('Error importing libraries or opening settings (prepare_data module)')
    print(ee1)


# functions definitions


def general_mean_scaler(local_array):
    if len(local_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_array, axis=1)
    mean_scaling = np.divide(local_array, 1 + mean_local_array)
    return mean_scaling


def window_based_normalizer(local_window_array):
    if len(local_window_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_window_array, axis=1)
    window_based_normalized_array = np.add(local_window_array, -mean_local_array)
    return window_based_normalized_array


def prepare():
    print('\n~prepare_data module~')
    # check if clean is done
    if local_script_settings['data_cleaning_done'] == "True":
        print('datasets already cleaned, based in settings info')
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' raw datasets already cleaned']))
        if local_script_settings['repeat_data_cleaning'] == "False":
            return True
        else:
            print('repeating data cleaning again')
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' cleaning raw datasets']))

    # preproccessing core
    try:
        # open raw_data
        raw_data_filename = 'sales_train_evaluation.csv'
        raw_data_sales = pd.read_csv(''.join([local_script_settings['raw_data_path'], raw_data_filename]))
        print('raw sales data accessed')

        # open sell prices and calendar data
        sell_prices = pd.read_csv(''.join([local_script_settings['raw_data_path'], 'sell_prices.csv']))
        calendar_data = pd.read_csv(''.join([local_script_settings['raw_data_path'], 'calendar.csv']))
        print('price and calendar data accessed')

        # extract and check correct data size
        print('loading and checking data..')
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
        print('check of data dimensions passed')

        # data general_mean based - rescaling
        nof_time_series = raw_unit_sales.shape[0]
        nof_selling_days = raw_unit_sales.shape[1]
        scaled_unit_sales = np.zeros(shape=(nof_time_series, nof_selling_days))
        for time_serie in range(nof_time_series):
            scaled_time_serie = general_mean_scaler(raw_unit_sales[time_serie: time_serie + 1, :])
            scaled_unit_sales[time_serie: time_serie + 1, :] = scaled_time_serie
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
        time_serie_group = []
        for time_serie in range(nof_time_series):
            time_serie_group.append([time_serie, 0])
            normalized_time_serie = []
            for window_start_day in window_first_days:
                window_array = scaled_unit_sales[
                               time_serie: time_serie + 1,
                               window_start_day: window_start_day + moving_window_length]
                normalized_window_array = window_based_normalizer(window_array)
                if window_start_day == last_window_start:
                    normalized_time_serie.append(normalized_window_array[-remainder_days:])
                else:
                    normalized_time_serie.append(normalized_window_array)
            exact_length_time_serie = np.array(normalized_time_serie).flatten()[: nof_selling_days]
            window_normalized_scaled_unit_sales[time_serie: time_serie + 1, :] = exact_length_time_serie
        print('data normalization done')

        # check if separating in groups was set
        nof_groups = local_script_settings['number_of_groups']
        if nof_groups > 1:
            # grouping in 3 major aggregations, load thresholds from settings
            group1_zero_sales_percentage_threshold = local_script_settings['group1_zero_sales_percentage_threshold']
            group2_zero_sales_percentage_threshold = local_script_settings['group2_zero_sales_percentage_threshold']
            group1_price_x_sale_quantile_threshold = local_script_settings['group1_price_x_sale_quantile_threshold']
            group2_price_x_sale_quantile_threshold = local_script_settings['group2_price_x_sale_quantile_threshold']

            # calculate dollar price x sales quantiles
            print('computing money-sales and applying threshold criteria for aggregation in three groups')
            sell_prices = np.array(sell_prices)
            raw_data_sales = np.array(raw_data_sales)
            price_x_sale = np.zeros(shape=(nof_time_series, local_settings_max_selling_time))
            calendar_weeks_ids = list(np.unique(calendar_data.iloc[:, 1].values))
            nof_weeks = len(calendar_weeks_ids)
            weeks_numbers = {calendar_weeks_ids[i]: i for i in range(nof_weeks)}
            days_from_week = {i: [i * 7 + j for j in range(7)] for i in range(nof_weeks + 1)}
            weeks_with_data = np.floor(local_settings_max_selling_time / 7).astype(int)
            nof_last_days = np.remainder(local_settings_max_selling_time, 7)
            last_days = [weeks_with_data * 7 + j for j in range(nof_last_days)]
            # this part of the code is explained in the dummy.txt file, in folder 1.1_documentation
            for week_sell_price in range(np.shape(sell_prices)[0]):
                if weeks_numbers[sell_prices[week_sell_price, 2]] > weeks_with_data:
                    continue
                elif weeks_numbers[sell_prices[week_sell_price, 2]] == weeks_with_data:
                    days = last_days
                else:
                    days = days_from_week[weeks_numbers[sell_prices[week_sell_price, 2]]]
                item_full_name = ''.join([sell_prices[week_sell_price, 1], '_',
                                          sell_prices[week_sell_price, 0], '_validation'])
                price_x_sale[np.where(raw_data_sales[:, 0] == item_full_name)[0][0], days[0]: days[-1] + 1] \
                    = sell_prices[week_sell_price, 3]
            price_x_sale = np.multiply(raw_unit_sales, price_x_sale)

            # separate in respective groups, according to the defined criteria
            group1_price_x_sale_quantile = np.quantile(price_x_sale[np.nonzero(price_x_sale)],
                                                       group1_price_x_sale_quantile_threshold)
            group2_price_x_sale_quantile = np.quantile(price_x_sale[np.nonzero(price_x_sale)],
                                                       group2_price_x_sale_quantile_threshold)
            group1, group2, group3, time_serie_group = [[] for groups in range(nof_groups + 1)]
            index_in_group1, index_in_group2, index_in_group3 = [0] * nof_groups
            list_of_index_group1, list_of_index_group2, list_of_index_group3 = [], [], []
            for time_serie in range(nof_time_series):
                time_serie_array = raw_unit_sales[time_serie, :]
                nof_zeros = max_selling_time - np.count_nonzero(time_serie_array)
                # meet criteria for G1, following current index convention, correspond to 0
                if (nof_zeros / max_selling_time) < group1_zero_sales_percentage_threshold and \
                        np.mean(price_x_sale[time_serie, :]) > group1_price_x_sale_quantile:
                    group1.append(window_normalized_scaled_unit_sales[time_serie, :])
                    time_serie_group.append([time_serie, 0])
                    list_of_index_group1.append([time_serie, index_in_group1])
                    index_in_group1 += 1
                # meet criteria for G2
                elif (nof_zeros / max_selling_time) > group2_zero_sales_percentage_threshold and \
                        np.mean(price_x_sale[time_serie, :]) < group2_price_x_sale_quantile:
                    group2.append(window_normalized_scaled_unit_sales[time_serie, :])
                    time_serie_group.append([time_serie, 1])
                    list_of_index_group2.append([time_serie, index_in_group2])
                    index_in_group2 += 1
                # if not in group1 neither group2, then meet criteria for G3
                else:
                    group3.append(window_normalized_scaled_unit_sales[time_serie, :])
                    time_serie_group.append([time_serie, 2])
                    list_of_index_group3.append([time_serie, index_in_group3])
                    index_in_group3 += 1
            group1 = np.array(group1)
            group2 = np.array(group2)
            group3 = np.array(group3)
            indexes_group1 = np.array(list_of_index_group1)
            indexes_group2 = np.array(list_of_index_group2)
            indexes_group3 = np.array(list_of_index_group3)
            indexes_in_groups = np.array([indexes_group1, indexes_group2, indexes_group3])
            np.save(''.join([local_script_settings['train_data_path'], 'price_x_sale']),
                    price_x_sale)
            np.save(''.join([local_script_settings['train_data_path'], 'indexes_in_groups']), indexes_in_groups)
        else:
            group1 = window_normalized_scaled_unit_sales
            group2 = np.array([])
            group3 = np.array([])
        time_serie_group = np.array(time_serie_group)

        # save clean data source for subsequent training
        np.save(''.join([local_script_settings['train_data_path'], 'group1']),
                group1)
        np.save(''.join([local_script_settings['train_data_path'], 'group2']),
                group2)
        np.save(''.join([local_script_settings['train_data_path'], 'group3']),
                group3)
        np.save(''.join([local_script_settings['train_data_path'], 'time_serie_group']),
                time_serie_group)
        np.save(''.join([local_script_settings['train_data_path'], 'x_train_source']),
                window_normalized_scaled_unit_sales)
        np.savetxt(''.join([local_script_settings['clean_data_path'], 'x_train_source.csv']),
                   window_normalized_scaled_unit_sales, fmt='%10.15f', delimiter=',', newline='\n')
        print('cleaned data -and their metadata- saved to file')
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' successful saved cleaned data and metadata']))
    except Exception as e1:
        print('Error at preproccessing raw data')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' data preproccesing error']))
        logger.error(str(e1), exc_info=True)
        return False

    # save settings
    try:
        if local_script_settings['metaheuristic_optimization'] == "False":
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['data_cleaning_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        elif local_script_settings['metaheuristic_optimization'] == "True":
            with open(''.join([local_script_settings['metaheuristics_path'],
                               'organic_settings.json']), 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['data_cleaning_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' settings modified and saved']))
        print('raw datasets cleaned, settings saved..')
    except Exception as e1:
        print('Error saving settings')
        print(e1)
        logger.error(str(e1), exc_info=True)

    # back to main code
    return True
