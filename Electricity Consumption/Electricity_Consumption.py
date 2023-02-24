import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA

from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Load the data into a dataframe (df).
df = pd.read_csv(r'C:\Users\patri\Desktop\Python\Data\Electric_Production.csv')
df.rename(columns = {'Value':'Energy Production'}, inplace = True)

# First and last 5 entries in the dataset.
print(df.head())
print(df.tail())
print(df.dtypes)

# Change the date to date time objects.
df['DATE'] = pd.to_datetime(df['DATE'])

print(df['DATE'].head())
print(df['DATE'].dt.year.head())

df = df.set_index('DATE')
print(df.dtypes)

# Train/test data split.
test_size = 36
train, test = df[:-test_size], df[-test_size:]

# Plotting the data and histogram.
fig, ax = plt.subplots(1,2)
ax[0].plot(train['Energy Production'])
ax[0].set(xlabel='Date', ylabel = 'Energy Production')

sns.histplot(train['Energy Production'], ax = ax[1], stat='density', kde=True)
ax[1].set(title = "Histogram & estimated density", xlabel = "Energy Production")

mu, std = stats.norm.fit(train["Energy Production"])
min, max = plt.xlim()
x = np.linspace(min, max, 100)
p = stats.norm.pdf(x, mu, std)
ax[1].plot(x, p, color = 'orange', label = 'norm pdf')
ax[1].legend(loc = 'best')
plt.show()

# Box-Cox tranformation test.
transformed, lam = stats.boxcox(train['Energy Production'].values)
print("Lambda value: %f" % lam)

# Start and end dates. 
start_date = datetime(2009, 1, 1)
end_date = datetime(2010, 12, 1)

# Look for seasonality and/or trend.
df[(start_date <= df.index) & (df.index <= end_date)].plot(grid = 'on')
plt.show()

# Dickey-Fuller test for stationarity.
def adfuller_test(data):
    res = adfuller(data)
    labels = ['ADF Statistic', 'p-value', '#lags uesd', 'Number of Observations Used']
    for value, label in zip(res, labels):
        print(label + ': ' + str(value))
    if res[1] <= 0.05:
        print('Strong evidence against the null hypothesis, reject the null hypothesis. Data is stationary.')
    else:
        print('Weak evidence against null hypothesis, time series has a unit root. Data is non-stationary.')

adfuller_test(train["Energy Production"])

# Look at seasonality/trend.
decomposition = sm.tsa.seasonal_decompose(df, model = "additive")
decomposition.plot()
plt.show()

# Remove seasonlity.
train['Seasonal Difference'] = train['Energy Production'] - train['Energy Production'].shift(12)
print(train.head(13))

# Dickey-Fuller test again.
adfuller_test(train["Seasonal Difference"].dropna())

# Start the Box-Jenkins method

# Correlation plots.
fig, ax = plt.subplots(1,2,figsize=(10,5))
plot_acf(train['Seasonal Difference'].dropna(), ax = ax[0], lags = 36)
plot_pacf(train['Seasonal Difference'].dropna(), ax = ax[1], lags = 36, method = 'ywm')
plt.show()

"""""
PACF cuts off at lag 6 and ACF cuts off at lag 21 - no real geometric decay.

Possible models -> AR(4), MA(21).

"""""
