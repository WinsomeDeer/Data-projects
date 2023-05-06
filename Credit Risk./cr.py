import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Load the data.
df  = pd.read_csv(r'C:\Users\patri\Desktop\Python\Data\credit_risk_dataset.csv')
print(df.describe())

# Check the data for NaN values.
print(df.isnull().sum())

# Have a considerable amount of NaNs in 2 of the columns - check the distributions.
fig, ax = plt.subplots(1,2)
ax[0].set_xlabel('Employment length')
ax[0].set_ylabel('Frequency')
sns.histplot(df['person_emp_length'], ax = ax[0])
ax[1].set_xlabel('Loan interest rate')
ax[1].set_ylabel('Frequency')
sns.histplot(df['loan_int_rate'], ax = ax[1])
plt.show()

# Data is right skewed - so need to replace the NaNs with median.
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace = True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace = True)

# Clearly some outliers regarding employment length and age.
clean_df = df[df['person_age'] <= 100]
clean_df = clean_df[clean_df['person_emp_length'] <= 50]
print(clean_df.describe())

# Check for NaNs.
print(df.isnull().sum())
