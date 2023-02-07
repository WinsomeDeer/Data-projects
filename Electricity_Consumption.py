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
start_date = datetime(1985, 1, 1)
end_time = datetime(2018, 1, 1)
