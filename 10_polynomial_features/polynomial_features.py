import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import metrics
from sklearn.pipeline import Pipeline

# خواندن داده‌ها
df = pd.read_csv('real_estate.csv')

# ---------- اضافه کردن بخش تحلیل داده ----------
# نمایش ماتریس همبستگی
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Pairplot برای بررسی روابط دوطرفه
sns.pairplot(df, vars=df.columns[:-1], diag_kind='kde')
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()
# ---------------------------------------------

# جدا کردن ویژگی‌ها و هدف
X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# ایجاد Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    # ('model', LinearRegression()),
    ('model', Ridge(alpha=1.0)),
])

# آموزش مدل
pipeline.fit(X_train, y_train)

# داده ورودی (بدون Y)
input_data = [[101, 2013.500, 17.5, 964.7496, 4, 24.98872, 121.53411]]
# تبدیل به DataFrame (با نام ستون‌های اصلی)
columns = df.drop('Y house price of unit area', axis=1).columns
df_input = pd.DataFrame(input_data, columns=columns)
predicted_price = pipeline.predict(df_input)
print(f"Predicted Price: {predicted_price[0]:.2f}")
actual_price = 38.2
difference = actual_price - predicted_price[0]
print(f"Actual Price: {actual_price}")
print(f"Difference: {difference:.2f}")


y_pred = pipeline.predict(X_test)

# ---------- اضافه کردن نمودارهای ارزیابی ----------
# 1. نمودار مقایسه y_test و y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Polynomial Regression)')
plt.show()

# 2. نمودار باقیمانده‌ها
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, bins=30)
plt.axvline(0, color='r', linestyle='--')
plt.title('Residuals Distribution')
plt.show()

# 3. نمودار رگرسیون چندجمله‌ای برای یک ویژگی (مثال: X2 house age)
plt.figure(figsize=(10, 6))
feature = 'X2 house age'
X_single = df[[feature]]
y_single = df['Y house price of unit area']

# آموزش مدل ساده برای نمایش منحنی
pipeline_single = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])

pipeline_single.fit(X_single, y_single)
X_plot = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
y_plot = pipeline_single.predict(X_plot)

plt.scatter(X_single, y_single, alpha=0.5, label='Data')
plt.plot(X_plot, y_plot, 'r', lw=2, label='Polynomial Fit')
plt.xlabel(feature)
plt.ylabel('House Price')
plt.title(f'Polynomial Regression ({feature})')
plt.legend()
plt.show()
# ---------------------------------------------

# محاسبه معیارها
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

metrics_df = pd.DataFrame({
    'MAE': [mae],
    'MSE': [mse],
    'R2': [r2]
})
print("Metrics:\n", metrics_df)

