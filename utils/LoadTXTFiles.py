# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
import pandas as pd
import os
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
# Augmented Dickey-Fuller Test (ADF Test)/unit root test
from statsmodels.tsa.stattools import adfuller
import seaborn as sb
import matplotlib.pyplot as plt

to_remove = ['ID', 'deviceid', 'line', 'Date', 'Time',
             'hour', 'minute', 'second', 'BEECNT_message OUT', 'BEECNT_message IN']

to_sum = ['BEECNT_message OUT', 'BEECNT_message IN']


def adf_test(ts, signif=0.05):
    dftest = adfuller(ts, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags', '# Observations'])
    for key, value in dftest[4].items():
        adf['Critical Value (%s)' % key] = value
    print(adf)
    p = adf['p-value']
    if p <= signif:
        print(f" Series is Stationary")
    else:
        print(f" Series is Non-Stationary")


def load_day(filename):
    BeePath = os.path.join('data_play', filename)
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


# data1 = load_day('BW201010.txt')
# data2 = load_day('BW201011.txt')
# data3 = load_day('BW201012.txt')
# data4 = load_day('BW201013.txt')
# data5 = load_day('BW201014.txt')
# data6 = load_day('BW201015.txt')
# data7 = load_day('BW201016.txt')
data8 = load_day('BW201017.txt')
data9 = load_day('BW201018.txt')
data10 = load_day('BW201019.txt')
data11 = load_day('BW201020.txt')
data12 = load_day('BW201021.txt')

data13 = load_day('BW201022.txt')
data14 = load_day('BW201023.txt')
data15 = load_day('BW201024.txt')
data16 = load_day('BW201025.txt')
data17 = load_day('BW201026.txt')
data18 = load_day('BW201027.txt')
data19 = load_day('BW201028.txt')
data20 = load_day('BW201029.txt')
data21 = load_day('BW201030.txt')
data22 = load_day('BW201031.txt')


frames = [data8, data9, data10,
          data11, data12, data13, data14, data15, data16, data17, data18, data19, data20,
          data21, data22]

result = pd.concat(frames, sort=False)
# del data1, data2, data3
# del data4, data5, data6, data7
del data8, data9, data10
del data11, data12, data13, data14, data15, data16, data17, data18, data19, data20,\
    data21, data22

# handle date
result['hour'] = result['hour']  # + ':00:00'
# result['Date_Time'] = pd.to_datetime(result.Date.str.lstrip(' ') + ' ' + result.hour.str.lstrip(' '),
#                                      format='%Y-%m-%d %H.%M.%S')
result.to_csv("data_play/bee_15.csv", index=False)

# --------------------------------------------------------------------------------------------------------
result.index = result.Date_Time
result = result.drop(['Date_Time', 'Date', 'hour'], axis=1)


# Visualize the trends in data_play ---------------------------------------------------
sb.set_style('darkgrid')
result.plot(kind='line', legend='reverse', title='Visualizing Sensor Array Time-Series')
# plt.legend(loc='upper right', shadow=True, bbox_to_anchor=(1.35, 0.8))
plt.show()

# Dropping Temperature & Relative Humidity as they do not change with Time
result.drop(['Temperature', 'Rel_Humidity'], axis=1, inplace=True)

# Again Visualizing the time-series data_play
sb.set_style('darkgrid')
result.plot(kind='line', legend='reverse', title='Visualizing Sensor Array Time-Series')
plt.legend(loc='upper right', shadow=True, bbox_to_anchor=(1.35, 0.8))
plt.show()


# Splitting the dataset into train & test subsets
ds_train = result[:int(0.8*(len(result)))]
ds_test = result[int(0.8*(len(result))):]

# Augmented Dickey-Fuller Test (ADF Test) to check for stationarity
# https://towardsdatascience.com/simple-multivariate-time-series-forecasting-7fa0e05579b2


def adf_test(ds):
    dftest = adfuller(ds, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags', '# Observations'])

    for key, value in dftest[4].items():
        adf['Critical Value (%s)' % key] = value
    print(adf)

    p = adf['p-value']
    if p <= 0.05:
        print("\nSeries is Stationary")
    else:
        print("\nSeries is Non-Stationary")


for i in ds_train.columns:
    print("Column: ", i)
    print('--------------------------------------')
    adf_test(ds_train[i])
    print('\n')

# Differencing all variables to get rid of Stationarity
ds_differenced = ds_train.diff().dropna()

# Running the ADF test once again to test for Stationarity
for i in ds_differenced.columns:
    print("Column: ", i)
    print('--------------------------------------')
    adf_test(ds_differenced[i])
    print('\n')

# Now cols: 3, 5, 6, 8 are non-stationary
ds_differenced = ds_differenced.diff().dropna()

# Running the ADF test for the 3rd time to test for Stationarity
for i in ds_differenced.columns:
    print("Column: ", i)
    print('--------------------------------------')
    adf_test(ds_differenced[i])
    print('\n')

# Now some cols are non-stationary
ds_differenced = ds_differenced.diff().dropna()

# Running the ADF test for the 3rd time to test for Stationarity
for i in ds_differenced.columns:
    print("Column: ", i)
    print('--------------------------------------')
    adf_test(ds_differenced[i])
    print('\n')

# non-stationary series
ds_differenced = ds_differenced.drop(['SW420_Vibrate HIVE'], axis=1)


# # add chunkID --------------------------------------------------------------------
# result["chunkID"] = pd.Series(np.ones(result.shape[0], np.int8))
# result["position_within_chunk"] = np.arange(1, result.shape[0] + 1)
# # export to csv
# result.to_csv(r'data_play/bee_10.csv', index=False, header=True)

# pyplot.plot(result['BEECNT_message OUT']) ---------------------------------------
result = result.set_index(result['Date'] + " " + result['hour'])
series = pd.Series(result['BEECNT_message OUT'])

series.plot()
pyplot.show()
autocorrelation_plot(series)
pyplot.show()

# split into train and test sets -------------------------------
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# walk-forward validation --------------------------------------
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# evaluate forecasts ------------------------------------------
rmse = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


