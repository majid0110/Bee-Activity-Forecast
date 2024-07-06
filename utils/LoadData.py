import numpy as np
from pandas import read_csv
import pandas as pd
import os


class DataLoader:

    # default constructor
    def __init__(self, data_path, file_name, model_output, sqrt_transform):
        self.data_path = data_path
        self.file_name = file_name
        self.model_output = model_output
        self.sqrt_transform = sqrt_transform

    def load_data(self):
        bee_df = read_csv(self.data_path + self.file_name, header=0, parse_dates=[['Date', 'hour']])

        bee_df['hour'] = pd.to_datetime(bee_df['Date_hour']).dt.hour
        bee_df['Day_part'] = np.where((bee_df['hour'] >= 18) | (bee_df['hour'] < 5), 0, 1)
        bee_df = bee_df.drop(['hour'], axis=1)

        # clean outlier
        bee_df.loc[bee_df.Day_part == 0, self.model_output] = 0

        # select columns for modeling *****
        bee_df = bee_df[['Date_hour', self.model_output, 'AM2302_1_Temp', 'AM2302_1_Humi',
                         'AM2302_2_Temp HIVE', 'AM2302_2_Humi HIVE', 'MHRD_rain', 'MQ135_PPM',
                         'BH1750_lux', 'VEML6750_uvindex', 'Day_part']]

        bee_df = bee_df.rename(columns={'Date_hour': 'ds', self.model_output: 'y',
                                        'AM2302_1_Temp': 'outTemp', 'AM2302_1_Humi': 'outHumi',
                                        'AM2302_2_Temp HIVE': 'hiveTemp', 'AM2302_2_Humi HIVE': 'hiveHumi',
                                        'MHRD_rain': 'rain', 'MQ135_PPM': 'ppm',
                                        'BH1750_lux': 'lux', 'VEML6750_uvindex': 'uvindex', 'Day_part': 'Day_night'})

        # create dataframe
        bee_df.reset_index(drop=True, inplace=True)
        # bee_df.set_index('ds', inplace=True, drop=True)

        if self.sqrt_transform:
            bee_df['y'] = np.sqrt(bee_df['y'])

        return bee_df