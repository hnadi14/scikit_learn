import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# خواندن داده‌ها
df = pd.read_csv('breast_cancer.csv')

# 1. پیش‌پردازش داده‌ها
# حذف ستون غیرضروری id
df.drop('id', axis=1, inplace=True)

# تبدیل برچسب‌ها به 0 و 1 (Malignant=1, Benign=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# ------------------------------ جدید: تشخیص و حذف داده‌های پرت ------------------------------
# محاسبه آماره‌های IQR برای ویژگی‌های انتخابی
Q1_radius = df['radius_mean'].quantile(0.25)
Q3_radius = df['radius_mean'].quantile(0.75)
IQR_radius = Q3_radius - Q1_radius

Q1_texture = df['texture_mean'].quantile(0.25)
Q3_texture = df['texture_mean'].quantile(0.75)
IQR_texture = Q3_texture - Q1_texture

Q1_perimeter_mean = df['perimeter_mean'].quantile(0.25)
Q3_perimeter_mean = df['perimeter_mean'].quantile(0.75)
IQR_perimeter_mean = Q3_perimeter_mean - Q1_perimeter_mean

# تعیین محدوده مجاز برای داده‌ها
lower_radius = Q1_radius - 1.5 * IQR_radius
upper_radius = Q3_radius + 1.5 * IQR_radius

lower_texture = Q1_texture - 1.5 * IQR_texture
upper_texture = Q3_texture + 1.5 * IQR_texture

lower_perimeter_mean = Q1_perimeter_mean - 1.5 * IQR_perimeter_mean
upper_perimeter_mean = Q3_perimeter_mean + 1.5 * IQR_perimeter_mean

# فیلتر کردن داده‌های پرت
filtered_data = df[
    (df['radius_mean'].between(lower_radius, upper_radius)) &
    (df['texture_mean'].between(lower_texture, upper_texture)) &
    (df['perimeter_mean'].between(lower_perimeter_mean, upper_perimeter_mean))
    ]

print(f"\nتعداد داده‌های پرت حذف شده: {len(df) - len(filtered_data)}")
# ------------------------------------------------------------------------------------------

# بررسی وجود مقادیر گم‌شده
print("\nتعداد مقادیر گم‌شده در هر ستون پس از حذف داده‌های پرت:")
print(filtered_data.isnull().sum())

# انتخاب ویژگی‌ها و متغیر هدف پس از حذف پرت‌ها
features = filtered_data[['radius_mean', 'texture_mean', 'perimeter_mean']]
target = filtered_data['diagnosis']

# 2. استانداردسازی ویژگی‌ها
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. تقسیم داده‌ها به مجموعه آموزشی و آزمون
x_train, x_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.33, random_state=42
)

# 4. آموزش مدل رگرسیون لجستیک
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)

# 5. پیش‌بینی و ارزیابی مدل
y_pred = log_reg.predict(x_test)
y_pred_proba = log_reg.predict_proba(x_test)[:, 1]

# محاسبه معیارهای ارزیابی
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# نمایش نتایج
print(f"\nدقت مدل پس از حذف داده‌های پرت: {accuracy:.2f}")
print("\nماتریس سردرگمی:")
print(conf_matrix)
print("\nگزارش طبقه‌بندی:")
print(classification_rep)
print(f"AUC-ROC: {roc_auc:.2f}")

# 6. مثال داده تست دستی برای پیش‌بینی
sample_data = pd.DataFrame([[11.0, 29.0, 70]],
                           columns=['radius_mean', 'texture_mean', 'perimeter_mean'])  # تغییر این خط

# استانداردسازی داده تست با استفاده از scaler آموزش دیده
sample_data_scaled = scaler.transform(sample_data)

# پیش‌بینی کلاس و احتمالات
sample_pred = log_reg.predict(sample_data_scaled)
sample_pred_proba = log_reg.predict_proba(sample_data_scaled)[0][1]

# نمایش نتیجه
print("\n\nمثال پیش‌بینی دستی:")
print(f"داده ورودی (استاندارد نشده): {sample_data.values[0]}")
print(f"داده استاندارد شده: {sample_data_scaled[0]}")
print(f"پیش‌بینی: {'بدخیم (M)' if sample_pred[0] == 1 else 'خوش‌خیم (B)'}")
print(f"احتمال بدخیمی: {sample_pred_proba:.2f}")


print(features_scaled)