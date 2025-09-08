import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import os
import matplotlib.pyplot as plt

# ------------------------------
# 1. بارگذاری داده‌ها
# ------------------------------
def load_data(file_path):
    """
    بارگذاری داده‌ها از فایل CSV.
    """
    try:
        df = pd.read_csv(file_path)
        print("داده‌ها با موفقیت بارگذاری شدند.")
        return df
    except Exception as e:
        print(f"خطا در بارگذاری داده‌ها: {e}")
        raise

# ------------------------------
# 2. پیش‌پردازش داده‌ها
# ------------------------------
def preprocess_data(df):
    """
    جداسازی ویژگی‌ها و متغیر هدف.
    """
    x = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']
    return x, y

# ------------------------------
# 3. نمونه‌گیری 15 درصد از داده‌ها
# ------------------------------
def sample_data(x, y, sample_size=0.15, random_state=42):
    """
    نمونه‌گیری تصادفی از داده‌ها با حفظ نسبت کلاس‌ها.
    """
    x_sampled, _, y_sampled, _ = train_test_split(
        x, y, train_size=sample_size, random_state=random_state, stratify=y
    )
    print(f"Sampled Data Shape: {x_sampled.shape}, Sampled Labels Shape: {y_sampled.shape}")
    return x_sampled, y_sampled

# ------------------------------
# 4. تقسیم داده‌ها به آموزش و تست
# ------------------------------
def split_data(x, y, train_size=0.10, test_size=0.05, random_state=42):
    """
    تقسیم داده‌ها به آموزش و تست با حفظ نسبت کلاس‌ها.
    """
    total_size = train_size + test_size
    x_train_test, _, y_train_test, _ = train_test_split(
        x, y, train_size=total_size, random_state=random_state, stratify=y
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_train_test, y_train_test, train_size=train_size / total_size,
        test_size=test_size / total_size, random_state=random_state, stratify=y_train_test
    )
    print(f"Train Data Shape: {x_train.shape}, Test Data Shape: {x_test.shape}")
    return x_train, x_test, y_train, y_test

# ------------------------------
# 5. ایجاد Pipeline
# ------------------------------
def create_pipeline():
    """
    ایجاد یک Pipeline برای استانداردسازی، PCA و آموزش مدل Logistic Regression.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # استانداردسازی داده‌ها
        ('pca', PCA()),  # PCA (تعداد مؤلفه‌ها بهینه‌سازی می‌شود)
        ('logistic', LogisticRegression(solver='saga', max_iter=1000))  # Logistic Regression
    ])
    return pipeline

# ------------------------------
# 6. ارزیابی مدل
# ------------------------------
def evaluate_model(pipeline, X_test, y_test):
    """
    ارزیابی مدل با معیارهای مختلف.
    """
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # نمودار Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.named_steps['logistic'].classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# ------------------------------
# 7. تجسم تأثیر PCA
# ------------------------------
def visualize_pca(pca, X_train_scaled):
    """
    تجسم Cumulative Explained Variance برای PCA.
    """
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA: Cumulative Explained Variance")
    plt.grid(True)
    plt.show()

# ------------------------------
# 8. مدیریت عدم تعادل داده‌ها
# ------------------------------
def handle_imbalance(X_train, y_train):
    """
    مدیریت عدم تعادل داده‌ها با استفاده از SMOTE.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled Data Shape: {X_resampled.shape}, Resampled Labels Shape: {y_resampled.shape}")
    return X_resampled, y_resampled

# ------------------------------
# 9. جستجوی پارامترهای بهینه
# ------------------------------
def optimize_pipeline(pipeline, X_train, y_train):
    """
    جستجوی پارامترهای بهینه با استفاده از GridSearchCV.
    """
    param_grid = {
        'pca__n_components': [10, 20, 30],  # تعداد مؤلفه‌های PCA
        'logistic__C': [0.1, 1, 10],  # تنظیم پارامتر C
        'logistic__solver': ['saga', 'lbfgs']  # الگوریتم حل
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    return grid_search.best_estimator_

# ------------------------------
# 10. اجرای کلی برنامه
# ------------------------------
if __name__ == "__main__":
    # ساخت مسیر نسبی و خواندن داده‌ها
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_folder = os.path.join(parent_dir, '24_dimensionality_reduction')
    file_path = os.path.join(data_folder, 'covtype.csv')

    # بارگذاری داده‌ها
    df = load_data(file_path)

    # پیش‌پردازش داده‌ها
    X, y = preprocess_data(df)

    # نمونه‌گیری 15 درصد از داده‌ها
    X_sampled, y_sampled = sample_data(X, y, sample_size=0.15)

    # تقسیم داده‌ها به آموزش و تست
    X_train, X_test, y_train, y_test = split_data(X_sampled, y_sampled, train_size=0.10, test_size=0.05)

    # مدیریت عدم تعادل داده‌ها
    X_train_resampled, y_train_resampled = handle_imbalance(X_train, y_train)

    # ایجاد Pipeline
    pipeline = create_pipeline()

    # بهینه‌سازی Pipeline
    optimized_pipeline = optimize_pipeline(pipeline, X_train_resampled, y_train_resampled)

    # آموزش مدل با داده‌های بهینه‌سازی‌شده
    optimized_pipeline.fit(X_train_resampled, y_train_resampled)

    # ارزیابی مدل
    evaluate_model(optimized_pipeline, X_test, y_test)

    # تجسم تأثیر PCA
    visualize_pca(optimized_pipeline.named_steps['pca'], X_train_resampled)

    # پیش‌بینی روی داده تست
    test_x = X.iloc[1:2]
    prediction = optimized_pipeline.predict(test_x)
    print(f"\nPrediction for test sample: {prediction}")