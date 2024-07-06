from matplotlib import pyplot

# prepare data_play for lstm
import numpy as np
from numpy import concatenate

from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from utils.LoadData import DataLoader

# ------------------------------------
# Config -----------------------------
# ------------------------------------

data_path = 'data/'
file_name = 'bee_October_2021.csv'
model_output = 'BEECNT_message IN'
sqrt_transform = True

# ------------------------------------
# Read in the data_play --------------
# ------------------------------------

DataLoader = DataLoader(data_path, file_name, model_output, sqrt_transform)
bee_df = DataLoader.load_data()
bee_df.set_index('ds', inplace=True, drop=True)

# PLOT DATA ------------------------------------------
# specify columns to plot
values = bee_df.values
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()

for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(bee_df.columns[group], y=0.5, loc='right')
    i += 1

pyplot.show()


# LSTM Data Preparation *************************************

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset ==========================
dataset = bee_df
values = bee_df.values
# ensure all data_play is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
scaled = scaler.transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[11, 12, 13, 14, 15, 16, 17, 18, 19]], axis=1, inplace=True)
print(reframed.head(5))

# Define and Fit Model =====================

# split into train and test sets
values = reframed.values
# select the number of days for the training phase (out of 20)
days = 18
n_train_hours = days * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network ========================================================
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=20, validation_data=(test_X, test_y),
                    verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction ========================================
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# calculate RMSE
if sqrt_transform:
    rmse = np.sqrt(mean_squared_error(np.power(inv_y, 2), np.power(inv_yhat, 2)))
else:
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
