import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ۱. بارگذاری داده خوشه‌بندی شده
df = pd.read_csv('Clustered_Videos.csv')

# ۲. انتخاب ویژگی‌ها و برچسب خوشه
# قبل از آموزش مدل، نوع داده X را به float32 تغییر دهید
X = df[['Calculated_Views', 'Duration_in_Seconds']].astype(np.float32)
y = df['ClusterLabel']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ۳. تبدیل برچسب‌های متنی به عدد (اگر لازم است)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ۴. آموزش مدل درخت تصمیم
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_scaled, y_encoded)

# ۵. نمایش درخت تصمیم
plt.figure(figsize=(20,10))
plot_tree(
    clf,
    filled=True,
    feature_names=['Views', 'Duration'],
    class_names=label_encoder.classes_,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree for Cluster Interpretation')
plt.show()

# ... (بخش‌های قبلی کد بدون تغییر)

# ۶. ذخیره قوانین درخت (اصلاح شده)
# حذف حلقه و استفاده مستقیم از apply
df['Decision Path'] = clf.apply(X_scaled)

# ذخیره داده
df.to_csv('Decision_Tree_Rules.csv', index=False)


# # ۶. ذخیره قوانین درخت
# rules = []
# # در حلقه for مربوط به استخراج قوانین
# for path in clf.decision_path(X_scaled).indices:
#     # تغییر کلیدی: استفاده از ایندکس مستقیم NumPy
#     sample = X_scaled[path].reshape(1, -1).astype(np.float32)
#     rules.append(clf.tree_.apply(sample)[0])
#
# df['Decision Path'] = rules
# df.to_csv('Decision_Tree_Rules.csv', index=False)