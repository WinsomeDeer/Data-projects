import statsmodels.formula.api as smf
import statsmodels.api as sma
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
df = df.set_index(df['DATE'])

# Plotting the data.
df.plot(grid = 'on', x = 'DATE', y = 'IPG2211A2N')
plt.show()

# Start and end dates. 
start_date = datetime(2009, 1, 1)
end_date = datetime(2010, 12, 1)

df[(start_date <= df.index) & (df.index <= end_date)].plot(grid = 'on', x = 'DATE', y = 'Energy Production')
plt.show()

# Remove the seasonality.

decomposition = sm.tsa.seasonal_decompose(df, model = "additive")
decomposition.plot(grid = 'on', x = 'DATE', y = 'Energy Production')
plt.show()
