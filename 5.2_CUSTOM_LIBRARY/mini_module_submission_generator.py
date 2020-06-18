# generate a submission file according M5 accuracy competition
import pandas as pd
import numpy as np


class save_submission:

    def save(self, name_csv, local_forecasts, local_sg_settings):
        try:
            # using template (sample_submission)
            local_forecasts_data_frame = np.genfromtxt(''.join([local_sg_settings['raw_data_path'],
                                                               'sample_submission.csv']),
                                                       delimiter=',', dtype=None, encoding=None)
            local_forecasts_data_frame[1:, 1:] = local_forecasts
            pd.DataFrame(local_forecasts_data_frame).to_csv(''.join([local_sg_settings['submission_path'],
                                                                    name_csv]),
                                                            index=False, header=None)
            print(name_csv, ' saved')
        except Exception as mini_module_error:
            print('submission_generator mini_module error: ', mini_module_error)
            return False
        return True
