import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
import plotly.express as px
import plotly.graph_objects as go

# بارگذاری داده‌ها
try:
    df = pd.read_csv('processed_house_prices.csv')
except FileNotFoundError:
    raise FileNotFoundError("File 'processed_house_prices.csv' not found. Please check the file path.")

# بررسی ستون‌های مورد نیاز
required_columns = ['SqFt', 'Bedrooms', 'Bathrooms', 'Price']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Required columns {required_columns} are missing from the dataset.")

# جداسازی ورودی‌ها و خروجی
X = df[['SqFt', 'Bedrooms', 'Bathrooms']]  # ورودی‌ها (Features)
y = df['Price']  # خروجی (Target)

# بررسی وجود مقادیر گم‌شده
if X.isnull().values.any() or y.isnull().values.any():
    raise ValueError("The dataset contains missing values. Please handle them before proceeding.")

# تعریف و آموزش مدل رگرسیون خطی
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # آموزش مدل روی تمام داده‌ها

# اجرای اعتبارسنجی متقاطع
try:
    result = cross_validate(
        estimator=lin_reg,  # مدل
        X=X,                # ورودی‌ها
        y=y,                # خروجی
        cv=5,               # تعداد بخش‌ها
        scoring='neg_mean_absolute_error',  # معیار ارزیابی
        return_train_score=True  # برگرداندن امتیاز آموزش نیز
    )
except Exception as e:
    raise RuntimeError(f"An error occurred during cross-validation: {e}")

# چاپ نتایج اعتبارسنجی متقاطع
print("Cross-Validation Results:")
for key, value in result.items():
    if key == 'test_score':
        print(f"{key}: {-value.mean():.4f} (Mean Absolute Error)")
    else:
        print(f"{key}: {value}")

# تست دستی: پیش‌بینی قیمت یک خانه جدید
new_house = pd.DataFrame({
    'SqFt': [2000],       # مساحت خانه (Square Feet)
    'Bedrooms': [3],      # تعداد اتاق‌خواب
    'Bathrooms': [2]      # تعداد حمام‌ها
})

# پیش‌بینی قیمت خانه جدید
predicted_price = lin_reg.predict(new_house)
print(f"\nPredicted Price for the new house: ${predicted_price[0]:,.2f}")

# رسم نمودارهای تعاملی
print("\nGenerating interactive visualizations...")

# 1. نمودار تعاملی رابطه بین مساحت (SqFt) و قیمت
fig1 = px.scatter(
    df,
    x='SqFt',
    y='Price',
    title="Interactive Relationship between Square Footage and Price",
    labels={'SqFt': 'Square Footage (SqFt)', 'Price': 'Price ($)'},
    template="plotly_dark",
    trendline="ols",  # اضافه کردن خط رگرسیون خطی
    color_discrete_sequence=["#FFA500"]  # رنگ جذاب
)
fig1.update_layout(
    xaxis_title="Square Footage (SqFt)",
    yaxis_title="Price ($)"
)
fig1.show()

# 2. نمودار تعاملی رابطه بین تعداد اتاق‌خواب (Bedrooms) و قیمت
fig2 = px.box(
    df,
    x='Bedrooms',
    y='Price',
    title="Interactive Relationship between Number of Bedrooms and Price",
    labels={'Bedrooms': 'Number of Bedrooms', 'Price': 'Price ($)'},
    template="plotly_white",
    color='Bedrooms',  # رنگ‌بندی بر اساس تعداد اتاق‌خواب
    color_discrete_sequence=px.colors.qualitative.Pastel  # رنگ‌های جذاب
)
fig2.update_layout(
    xaxis_title="Number of Bedrooms",
    yaxis_title="Price ($)"
)
fig2.show()

# 3. نمودار تعاملی رابطه بین تعداد حمام‌ها (Bathrooms) و قیمت
fig3 = px.violin(
    df,
    x='Bathrooms',
    y='Price',
    title="Interactive Relationship between Number of Bathrooms and Price",
    labels={'Bathrooms': 'Number of Bathrooms', 'Price': 'Price ($)'},
    template="plotly_white",  # قالب تمیز و سفید
    color='Bathrooms',  # رنگ‌بندی بر اساس تعداد حمام‌ها
    color_discrete_sequence=px.colors.qualitative.Pastel,  # رنگ‌های جذاب Pastel
    box=True,  # نمایش Box داخل Violin
    points="all",  # نمایش تمام نقاط داده روی نمودار
    # opacity=0.8  # شفافیت برای زیبایی بیشتر
)
fig3.update_traces(
    meanline_visible=True,
    scalemode="width",
    line_color="black",
    line_width=1
)
fig3.update_layout(
    xaxis_title="Number of Bathrooms",
    yaxis_title="Price ($)",
    font=dict(size=14),  # اندازه فونت بزرگ‌تر
    plot_bgcolor="rgba(0,0,0,0)",  # زمینه شفاف
    paper_bgcolor="rgba(0,0,0,0)"  # زمینه کاغذ شفاف
)
fig3.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x=df['Bathrooms'], y=df['Price'])
plt.title("Relationship between Number of Bathrooms and Price", fontsize=16)
plt.xlabel("Number of Bathrooms", fontsize=14)
plt.ylabel("Price ($)", fontsize=14)
plt.grid(True)
plt.show()


# 4. نمودار تعاملی همبستگی بین ویژگی‌ها و قیمت
correlation_matrix = df[['SqFt', 'Bedrooms', 'Bathrooms', 'Price']].corr()
fig4 = go.Figure(
    data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',  # رنگ‌بندی جذاب
        zmin=-1, zmax=1,  # محدوده مقادیر
        colorbar=dict(title="Correlation")
    )
)
fig4.update_layout(
    title="Interactive Correlation Matrix",
    xaxis_title="Features",
    yaxis_title="Features",
    template="plotly_white"
)
fig4.show()

# 5. نمودار تعاملی توزیع قیمت خانه‌ها
fig5 = px.histogram(
    df,
    x='Price',
    nbins=30,
    title="Interactive Distribution of House Prices",
    labels={'Price': 'Price ($)'},
    marginal="box",  # نمایش یک نمودار Boxplot در کنار Histogram
    template="plotly_dark",
    color_discrete_sequence=["#00FFFF"]  # رنگ جذاب
)
fig5.update_layout(
    xaxis_title="Price ($)",
    yaxis_title="Frequency",
    bargap=0.1  # فاصله بین میله‌ها
)
fig5.show()