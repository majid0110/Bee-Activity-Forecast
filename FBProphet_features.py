import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from utils.LoadData import DataLoader

# ------------------------------------
# Config -----------------------------
# ------------------------------------

data_path = 'data/'
file_name = 'bee_October_2021.csv'
model_output = 'BEECNT_message IN'
sqrt_transform = True

# read data into dataframe
DataLoader = DataLoader(data_path, file_name, model_output, sqrt_transform)
df = DataLoader.load_data()


# ------------------------------------
# Plot data --------------------------
# ------------------------------------
# plt.rcParams['figure.figsize'] = (20, 10)
# plt.style.use('ggplot')
# pd.plotting.register_matplotlib_converters()
#
# plot = df.set_index('ds').y.plot()
# plot.set_xlabel('Days')
# plot.set_ylabel('Number of bee outings per hour')
# plot.get_figure()

# --------------------------------------
# TRAIN Prophet model - SINGLE MODEL ---
# --------------------------------------

model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True)
model.add_regressor('outTemp')
model.add_regressor('outHumi')
# new features
model.add_regressor('hiveTemp')
model.add_regressor('hiveHumi')
model.add_regressor('rain')
model.add_regressor('ppm')
model.add_regressor('lux')
model.add_regressor('uvindex')
model.add_regressor('Day_night')

# fit the model
model.fit(df)

# predict
forecast = model.predict(df.drop(columns="y"))
# fitter
forecast.loc[forecast.yhat < 0, "yhat"] = 0

model.plot(forecast)
model.plot_components(forecast)

# evaluate
metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
metric_df.dropna(inplace=True)

if sqrt_transform:
    np.sqrt(mean_squared_error(np.power(metric_df.y, 2), np.power(metric_df.yhat, 2)))
else:
    np.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))


# plot prediction ================================
model.plot(forecast)
model.plot_components(forecast)

# ---------------------------------------------------------------------------------------------------------------------
# SPLIT TRAIN AND TEST DATASET ----------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


def data_gen(df, tscv):
    """
    :param df: dataframe to split
    :param tscv: TimeSeriesSplit object
    :return: dataframe with with split labels
    """

    out_df = pd.DataFrame()
    for i, (train_i, test_i) in enumerate(tscv.split(df)):  # For Time Series Split
        # Use indexes to grab the correct data_play for this split
        train_df = df.copy().iloc[train_i, :]
        test_df = df.copy().iloc[test_i, :]

        # Combine predictions and training into one df for plotting
        train_df["train"] = "Train"
        test_df["train"] = "Test"

        sub_df = train_df.append(test_df).reset_index(drop=True)
        sub_df["split"] = "Split " + str(i + 1)

        out_df = out_df.append(sub_df).reset_index(drop=True)

    return out_df


# ------------------------------------
# TRAIN Prophet model ----------------
# ------------------------------------

tscv = TimeSeriesSplit(n_splits=5)
day_seas_df = data_gen(df, tscv)
rmses = []

for i in range(5):
    split_part = "Split " + str(i+1)
    split = day_seas_df[day_seas_df['split'] == split_part]

    # train
    df = split[split['train'] == "Train"]

    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True)

    model.add_regressor('outTemp')
    model.add_regressor('outHumi')
    # new features
    model.add_regressor('hiveTemp')
    model.add_regressor('hiveHumi')
    model.add_regressor('rain')
    model.add_regressor('ppm')
    model.add_regressor('lux')
    model.add_regressor('uvindex')
    model.add_regressor('Day_night')

    model.fit(df)

    # predict
    df = split[split['train'] == "Test"]
    forecast = model.predict(df.drop(columns="y"))

    metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)
    if sqrt_transform:
        rmse = np.sqrt(mean_squared_error(np.power(metric_df.y, 2), np.power(metric_df.yhat, 2)))
    else:
        rmse = np.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))

    rmses.append(rmse)


print(f"Average RMSE:{np.mean(rmses)}")
