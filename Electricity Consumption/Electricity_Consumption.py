import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Load the data into a dataframe (df).
df = pd.read_csv(r'C:\Users\patri\Desktop\Python\Data\Electric_Production.csv')
df.rename(columns = {'IPG2211A2N':'Energy Production'}, inplace = True)

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

# Plotting the data.
df.plot(grid = 'on')
plt.show()

# Start and end dates. 
start_date = datetime(2009, 1, 1)0
end_date = datetime(2010, 12, 1)

# Look for seasonality and/or trend.
df[(start_date <= df.index) & (df.index <= end_date)].plot(grid = 'on')
plt.show()

# Look at seasonality/trend.
decomposition = sm.tsa.seasonal_decompose(df, model = "additive")
decomposition.plot()
plt.show()

# Remove seasonlity.
df_adj = df.diff(periods = 12)
df_adj.plot()
plt.show()

# Remove trend.
df_adj2 = df.diff(periods = 1)
df_adj2.plot()
plt.show()

fig_cf, ax = plt.subplots(1,2)
ax[0,0]

plot_acf(df)
plt.show()
plot_pacf(df)
plt.show()
