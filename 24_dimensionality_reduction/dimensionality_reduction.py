import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, roc_curve,
    auc
)
import seaborn as sns
import matplotlib.pyplot as plt

# 1. بارگذاری داده‌ها
df = pd.read_csv('covtype.csv')
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# 2. انتخاب ویژگی‌های بر اساس همبستگی
corr_matrix = df.corr()
most_corr = np.abs(corr_matrix['Cover_Type']).sort_values(ascending=False).iloc[1:11].index.tolist()
X_corr = X[most_corr]

# 3. انتخاب ویژگی‌ها با SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_corr, y)

# تبدیل خروجی SelectKBest به DataFrame با نام ستون‌های اصلی
selected_features = X_corr.columns[selector.get_support()].tolist()
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# 4. حذف ویژگی‌های با واریانس پایین
var_threshold = VarianceThreshold(threshold=0.1)
X_high_var = var_threshold.fit_transform(X_selected_df)

# تبدیل خروجی VarianceThreshold به DataFrame با نام ستون‌های باقیمانده
final_features = X_selected_df.columns[var_threshold.get_support()].tolist()
X_final = pd.DataFrame(X_high_var, columns=final_features)

print("Features after Variance Thresholding:")
print(final_features)

# 5. تقسیم داده‌ها به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.3, random_state=42, stratify=y
)

# 6. تنظیم پارامترهای SVM با GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svc = SVC(probability=True, random_state=42)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# 7. ارزیابی مدل روی داده‌های آزمون
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# محاسبه معیارهای ارزیابی
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. تجسم Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9. تجسم ROC Curve
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_proba_bin = lb.transform(np.argmax(y_proba, axis=1))

fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()