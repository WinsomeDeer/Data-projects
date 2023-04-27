
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
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
