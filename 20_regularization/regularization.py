import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

# خواندن داده‌ها
df = pd.read_csv('processed_house_prices.csv')

# تعریف ویژگی‌ها و هدف
X = df[['SqFt', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# ---- 1. نمودار همبستگی بین ویژگی‌ها ----
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# ---- 2. مدل رگرسیون خطی ----
lin_reg = LinearRegression()
lin_result = cross_validate(lin_reg, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Linear Regression MAE: {-np.mean(lin_result['test_score']):.2f}")

# ---- 3. تنظیم alpha برای Ridge و Lasso با استفاده از GridSearchCV ----
alphas = [0.01, 0.1, 1, 10, 100]  # مقادیر مختلف برای آزمایش alpha

# Ridge
ridge_pipe = Pipeline([('ridge', Ridge())])
ridge_grid = GridSearchCV(ridge_pipe, param_grid={'ridge__alpha': alphas}, cv=5, scoring='neg_mean_absolute_error')
ridge_grid.fit(X, y)
print(f"Ridge Best Alpha: {ridge_grid.best_params_['ridge__alpha']}")
print(f"Ridge MAE: {-ridge_grid.best_score_:.2f}")

# Lasso
lasso_pipe = Pipeline([('lasso', Lasso(max_iter=10000))])  # max_iter افزایش داده شده برای همگرایی
lasso_grid = GridSearchCV(lasso_pipe, param_grid={'lasso__alpha': alphas}, cv=5, scoring='neg_mean_absolute_error')
lasso_grid.fit(X, y)
print(f"Lasso Best Alpha: {lasso_grid.best_params_['lasso__alpha']}")
print(f"Lasso MAE: {-lasso_grid.best_score_:.2f}")

# ---- 4. نمودار مقایسه عملکرد Ridge و Lasso برای مقادیر مختلف alpha ----
ridge_scores = [-np.mean(score) for score in ridge_grid.cv_results_['mean_test_score']]
lasso_scores = [-np.mean(score) for score in lasso_grid.cv_results_['mean_test_score']]

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_scores, label='Ridge', marker='o')
plt.plot(alphas, lasso_scores, label='Lasso', marker='o')
plt.xscale('log')  # مقیاس لگاریتمی برای alpha
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Comparison of Ridge and Lasso Performance')
plt.legend()
plt.grid(True)
plt.show()