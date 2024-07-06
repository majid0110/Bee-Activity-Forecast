import pandas as pd
import os
# from matplotlib import pyplot
# from pandas.plotting import autocorrelation_plot
# from statsmodels.tsa.stattools import adfuller
# import seaborn as sb
import matplotlib.pyplot as plt

to_remove = ['ID', 'deviceid', 'line', 'Date', 'Time',
             'hour', 'minute', 'second', 'BEECNT_message OUT', 'BEECNT_message IN']

to_sum = ['BEECNT_message OUT', 'BEECNT_message IN']


def load_day(filename):
    BeePath = os.path.join('data2', filename)
    data = pd.read_csv(BeePath, delimiter=',')
    time = data['Time'].str.split(':')
    time = pd.DataFrame(time.tolist())
    time.columns = ['hour', 'minute', 'second']
    data = data.join(time)

    # data_play grouping
    to_mean = list(data.columns)

    for c in to_remove:
        to_mean.remove(c)

    # grouping
    hour_mean = data.groupby(['hour'], as_index=False)[to_mean].mean()
    hour_sum = data.groupby(['hour'], as_index=False)[to_sum].sum()
    hour_sum = hour_sum.drop(columns=['hour'], axis=1)

    data = pd.concat([data['Date'][0:24], hour_mean, hour_sum], axis=1)

    return data


months = {'September': '09', 'October': '10', 'November': '11', 'December': '12'}
start_date = 1
take_days = 20
year = 'BW20'
frames = []
month = 'October'

for i in range(take_days):
    if len(str(start_date + i)) > 1:
        day = str(start_date + i)
    else:
        day = '0' + str(start_date + i)

    date = year + months[month] + day + '.txt'
    data = load_day(date)
    frames.append(data)


result = pd.concat(frames, sort=False)
# handle date
result['hour'] = result['hour'] + ':00:00'
# result['Date_Time'] = pd.to_datetime(result.Date.str.lstrip(' ') + ' ' + result.hour.str.lstrip(' '),
#                                      format='%Y-%m-%d %H.%M.%S')

# plot data_play *************************************************************

# plt.figure()
result.reset_index().plot(y="BEECNT_message OUT", use_index=True)
result.reset_index().plot(y="BEECNT_message IN", use_index=True)
plt.show()

# export to csv
filename = 'bee_' + month + '_' + str(take_days) + '.csv'
result.to_csv(filename, index=False)


# =====================================================================================
# data1 = load_day('BW201010.txt')
# data2 = load_day('BW201011.txt')
# data3 = load_day('BW201012.txt')
# data4 = load_day('BW201013.txt')
# data5 = load_day('BW201014.txt')
# data6 = load_day('BW201015.txt')
# data7 = load_day('BW201016.txt')

# data8 = load_day('BW201017.txt')
# data9 = load_day('BW201018.txt')
# data10 = load_day('BW201019.txt')
# data11 = load_day('BW201020.txt')
# data12 = load_day('BW201021.txt')
#
# data13 = load_day('BW201022.txt')
# data14 = load_day('BW201023.txt')
# data15 = load_day('BW201024.txt')
# data16 = load_day('BW201025.txt')
# data17 = load_day('BW201026.txt')
# data18 = load_day('BW201027.txt')
# data19 = load_day('BW201028.txt')
# data20 = load_day('BW201029.txt')
# data21 = load_day('BW201030.txt')
# data22 = load_day('BW201031.txt')


# frames = [data8, data9, data10, data11, data12, data13, data14, data15,
#           data16, data17, data18, data19, data20, data21, data22]
#
# result = pd.concat(frames, sort=False)
# # del data1, data2, data3
# # del data4, data5, data6, data7
# del data8, data9, data10
# del data11, data12, data13, data14, data15, data16, data17, data18, data19, data20,\
#     data21, data22

# # handle date
# result['hour'] = result['hour']  # + '.00.00'
# # result['Date_Time'] = pd.to_datetime(result.Date.str.lstrip(' ') + ' ' + result.hour.str.lstrip(' '),
# #                                      format='%Y-%m-%d %H.%M.%S')

# # plot data_play *****
#
# # plt.figure()
# result.reset_index().plot(y="BEECNT_message OUT", use_index=True)
# result.reset_index().plot(y="BEECNT_message IN", use_index=True)
# plt.show()
#
# # export to csv
# result.to_csv("bee_15.csv", index=False)
