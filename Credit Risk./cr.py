import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection,linear_model, metrics
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix, roc_auc_score, classification_report
import xgboost as xgb
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

# NaNs all removed.
print(df.isnull().sum())

# Retrieve the numerical columns only.
clean_num_cols = pd.DataFrame(clean_df[clean_df.select_dtypes(include = ['float', 'int']).columns])

# Find the correlation plot.
corr = clean_num_cols.corr().sort_values('loan_status', axis = 1, ascending = False)
corr = corr.sort_values('loan_status', axis = 0, ascending=True)
corr_arr = np.zeros_like(corr)
corr_arr[np.triu_indices_from(corr_arr, k = 1)] = True

fig, ax = plt.subplots()
ax = sns.heatmap(corr, mask = corr_arr, square = True, annot = True, center = 0, cmap='RdBu', annot_kws = {"size": 12})
plt.show()

# Need to covnert the categorical labels into numerical ones - one hot-encode.
cat_cols = pd.DataFrame(clean_df[clean_df.select_dtypes(include = ['object']).columns])
print(cat_cols.columns)

# One hot-encode.
encoded_pd = pd.get_dummies(cat_cols)
concat_df = pd.concat([encoded_pd, clean_df['loan_status']], axis = 1)

# Create the new correlation matrix.
corr = concat_df.corr().sort_values('loan_status', axis = 1, ascending = False)
corr = corr.sort_values('loan_status', axis = 0, ascending = True)
corr_arr = np.zeros_like(corr)
corr_arr[np.triu_indices_from(corr_arr, k = 1)] = True

fig, ax = plt.subplots(figsize = (20, 15))
ax = sns.heatmap(corr, mask = corr_arr, square = True, annot = True, center = 0, fmt='.2f', cmap='RdBu',annot_kws = {"size": 12})
plt.show()

clean_df = pd.concat([clean_num_cols, encoded_pd], axis = 1)
print(clean_df.head())

# Train, test data split.
response_var = clean_df['loan_status']
explanatory_vars = clean_df.drop('loan_status', axis = 1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(explanatory_vars, response_var, random_state = 10, test_size = .20)
print('The training set has {} entries and the test set has {}'.format(X_train.shape[0], X_test.shape[0]))

# Function to test the various models - Logistic reg., Decision tree, XGBoost.
def model_testing(model, name = 'model_name'):
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_train)
    print(name, '\n', classification_report(Y_test, model.predict(X_test)))

# Logisitic regression.
log_reg = LogisticRegression(random_state = 10)
model_testing(log_reg, 'Logisitic regression')

# Decision tree.
dec_tree = DecisionTreeClassifier(max_depth = 10, min_samples_split = 2, min_samples_leaf = 1, random_state = 10)
model_testing(dec_tree, 'Decision Tree')

# KNN model.
KNN = KNeighborsClassifier(n_neighbors = 150)
model_testing(KNN, 'KNN')

# XGBoost model.
XGB = xgb.XGBClassifier(objective = "binary:logistic", random_state = 10)
model_testing(XGB, 'XGBoost')
