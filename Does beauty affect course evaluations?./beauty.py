import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sma
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

# Load the data into a dataframe (df).
df = pd.read_csv(r"C:\Users\patri\Desktop\Python\Data\beauty2.csv")
print(df)

# Correlation Matrix and correlation plot
corr_matrix = df.corr()
print(corr_matrix)
sns.pairplot(df)
plt.show()

# The initial linear regression model.
f = 'score ~ rank + ethnicity + gender + language + age + cls_perc_eval + cls_students + cls_level + cls_profs + cls_credits + bty_avg + pic_outfit + pic_color'
y = smf.ols(formula = f, data = df)
Y = y.fit()
print(Y.summary())

# The second linear regression model.
f = 'score ~ rank + ethnicity + gender + language + age + cls_perc_eval + cls_students + cls_credits + bty_avg + pic_outfit + pic_color'
y = smf.ols(formula = f, data = df)
Y = y.fit()
print(Y.summary())

# The third linear regression model.
f = 'score ~ rank + ethnicity + gender + language + age + cls_perc_eval + cls_credits + bty_avg + pic_outfit + pic_color'
y = smf.ols(formula = f, data = df)
Y = y.fit()
print(Y.summary())

# Studentized residuals Plot.
res = Y.outlier_test()
x = df['score']
y = res['student_resid']

plt.scatter(x, y)
plt.axhline(y = 0, color = 'black', linestyle = '--')
plt.xlabel('Points')
plt.ylabel('Studentized Residuals')
plt.show()

# Normal QQ plot
sma.qqplot(Y.resid, line = '45')
plt.show()
