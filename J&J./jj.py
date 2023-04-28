from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from itertools import product
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Readt the data into a dataframe.
df = pd.read_csv(r'C:\Users\patri\Desktop\Python\Data\jj.csv')
print(df.head())

fig, ax = plt.subplots()

# Set the plot settings.
ax.plot(df.time, df.value)
ax.set_xlabel('Year')
ax.set_ylabel('Earnings per share (USD)')

# Adickey fuller test function.
def adfuller_test(data):
    res = adfuller(data)
    labels = ['ADF Statistic', 'p-value', '#lags uesd', 'Number of Observations Used']
    for value, label in zip(res, labels):
        print(label + ': ' + str(value))
    if res[1] <= 0.05:
        print('Strong evidence against the null hypothesis, reject the null hypothesis. Data is stationary.')
    else:
        print('Weak evidence against null hypothesis, time series has a unit root. Data is non-stationary.')

# Adickey Fuller test for data.
adfuller_test(df['value'])

# Difference the data once.
diff_data = np.diff(df['value'], n = 1)
adfuller_test(diff_data)

# Difference the data again.
diff_data = np.diff(df['value'], n = 2)
adfuller_test(diff_data)

def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int):
    
    res = []

    for order in order_list:
        try:
            model = SARIMAX(endog, order = (order[0], d, order[1]), simple_differencing = False).fit(disp = False)
        except:
            continue
        aic = model.aic
        res.append([order, aic])
    res_df = pd.DataFrame(res)
    res_df.columns = ['(p, q)', 'AIC']
    
    res_df = res_df.sort_values(by = 'AIC', ascending = True).reset_index(drop = True)

    return res_df

p = range(0, 4, 1)
q = range(0, 4, 1)
list_of_combos = list(product(p, q))
train_data = df['value'][:-4]

res_df = optimize_ARIMA(train_data, list_of_combos, 2)
print(res_df.head())

# Best model is fou8nd to be the ARIMA(3,2,3).
model = SARIMAX(train_data, order = (3,2,3), simple_differencing = False)
model_fit = model.fit(disp = False)
model_fit.plot_diagnostics(figsize = (10, 8))
plt.show()

# Forecast the data.
test_data = df.iloc[-4:]
test_data['Naive seasonal data'] = df['value'].iloc[76:80].values
pred_data = model_fit.get_prediction(80, 83).predicted_mean
test_data['ARIMA Pred'] = pred_data

print(test_data.head())
print(df.head())

# Forecast plotting.
fig, ax = plt.subplots()

ax.plot(df['value'])
ax.plot(test_data['value'], 'b-', label = 'actual')
ax.plot(test_data['Naive seasonal data'], 'k--', label = 'naive seasonal')
ax.plot(test_data['ARIMA Pred'], 'r:', label = 'ARIMA(3,2,3)')
ax.set_xlabel('Year')
ax.set_ylabel('Earnings per share (USD)')
ax.legend(loc = 2)
ax.axvspan(80, 83, color = '#808080', alpha = 0.2)
plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])
ax.set_xlim(60, 83)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# MAPE function.
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true))* 100

ARIMA_mape = mape(test_data['value'], test_data['ARIMA Pred'])
NS_mape = mape(test_data['value'], test_data['Naive seasonal data'])

print(ARIMA_mape)
print(NS_mape)
