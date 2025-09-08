import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# خواندن دادهها
df = pd.read_csv('mall_customers.csv')

# نمایش دادههای اولیه
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Annual_Income_(k$)', y='Spending_Score_(1-100)', data=df, color='blue')
plt.title('Original Data Distribution')
plt.show()

# آمادهسازی دادهها
X = df[['Annual_Income_(k$)', 'Spending_Score_(1-100)']]

# مدل Isolation Forest با تنظیمات بهینه
model = IsolationForest(
    n_estimators=100,       # تعداد درختان
    max_samples='auto',     # نمونهگیری خودکار
    contamination=0.05,     # 5% دادهها به عنوان outlier
    random_state=42
)

# آموزش مدل
model.fit(X)

# محاسبه امتیازها و برچسبها
df['anomaly_score'] = model.decision_function(X)
df['anomaly_label'] = model.predict(X)

# تعداد نقاط پرت
n_outliers = df[df['anomaly_label'] == -1].shape[0]
print(f"تعداد نقاط پرت شناسایی شده: {n_outliers}")

# رسم نمودار با برچسبهای anomaly
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x='Annual_Income_(k$)',
    y='Spending_Score_(1-100)',
    hue='anomaly_label',
    data=df,
    palette={1: 'green', -1: 'red'},  # رنگ سبز برای نرمال، قرمز برای پرت
    style='anomaly_label',            # شکل متفاوت برای پرتها
    markers=[',', 'X'],               # نقطه برای نرمال، X برای پرت
    s=100                             # اندازه مارکر
)
plt.title('Anomaly Detection with Isolation Forest')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Anomaly Label')
plt.show()

# نمایش توزیع امتیازها
plt.figure(figsize=(12, 6))
sns.histplot(df['anomaly_score'], kde=True, bins=30)
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.axvline(x=0, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()