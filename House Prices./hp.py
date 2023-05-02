import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from typing import Union
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Read the data into pd dataframe. 
df  = pd.read_csv(r'C:\Users\patri\Desktop\Python\Data\raw_sales.csv')
# Change the dtype of datesold.
df.index = pd.to_datetime(df.datesold)
print(df.head())

# Resample the data to regularize the data.
df_resample = df.resample('M').mean()
print(df_resample.head())
print(len(df_resample))

# Check for stationarity.
def adfuller_test(data):
    res = adfuller(data)
    labels = ['ADF Statistic', 'p-value', '#lags uesd', 'Number of Observations Used']
    for value, label in zip(res, labels):
        print(label + ': ' + str(value))
    if res[1] <= 0.05:
        print('Strong evidence against the null hypothesis, reject the null hypothesis. Data is stationary.')
    else:
        print('Weak evidence against null hypothesis, time series has a unit root. Data is non-stationary.')

# Check for price.
adfuller_test(df_resample['price'])

df_resample_diff = np.diff(df_resample['price'], n = 1)
adfuller_test(df_resample_diff)

# Plot the data.
fig, ax = plt.subplots(figsize = (20,5))
plt.style.use('fivethirtyeight')
ax.plot(df_resample['price'])
ax.set_xlabel('Date')
ax.set_ylabel('House Price (GBP)')
plt.show()

# Diagnostics of the data.
fig, ax = plt.subplots(1, 2)
sns.boxplot(df_resample['price'], ax = ax[0])
sns.histplot(df_resample['price'], ax = ax[1])
plt.show()

# No signs of skewness - no transform needed.
fig, ax = plt.subplots(1,2)
sns.boxplot(df_resample['price'], ax = ax[0])
sns.histplot(df_resample['price'], ax = ax[1])
plt.show()

# Data more even distributed now, find the best Time Series model.

# Function to iterate through the best possible models.
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

# Possible values for p & q.
p = range(0, 4, 1)
q = range(0, 4, 1)
d = 1
p_q_pairs = list(product(p, q))

model_df = optimize_ARIMA(df_resample['price'][:110], p_q_pairs, 1)

print(model_df.head())

# Best model for forecasting is the AR(1) (with differenced data).
best_model = SARIMAX(df_resample['price'], order = (1, 1, 0))
AR_one = best_model.fit()
print(AR_one.summary())

AR_one.plot_diagnostics()
plt.show()

df_resample['price predict'] = AR_one.predict(start = 110, end = 150)

fig, ax = plt.subplots(figsize = (20,5))
ax.plot(df_resample['price'])
ax.plot(df_resample['price predict'])
ax.set_xlabel('Date')
ax.set_ylabel('House Price (GBP)')
plt.show()
