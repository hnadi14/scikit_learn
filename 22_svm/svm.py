import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# تنظیمات لاگ‌گیری
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data():
    """
    بارگذاری و پیش‌پردازش داده‌ها.

    این تابع داده‌های Breast Cancer را از کتابخانه sklearn بارگذاری می‌کند و پیش‌پردازش‌های اولیه را انجام می‌دهد:
    1. بررسی وجود مقادیر گمشده و پر کردن آنها (در صورت وجود).
    2. تقسیم داده‌ها به دو بخش آموزش و آزمون.

    Returns:
        x_train (pd.DataFrame): داده‌های ویژگی‌ها برای آموزش.
        x_test (pd.DataFrame): داده‌های ویژگی‌ها برای آزمون.
        y_train (pd.Series): برچسب‌های هدف برای آموزش.
        y_test (pd.Series): برچسب‌های هدف برای آزمون.
    """
    try:
        # بارگذاری داده‌ها
        x, y = datasets.load_breast_cancer(as_frame=True, return_X_y=True)
        logging.info("داده‌ها با موفقیت بارگذاری شدند.")

        # بررسی وجود مقادیر گمشده
        if x.isnull().sum().sum() > 0:
            logging.warning("مقادیر گمشده در داده‌ها شناسایی شدند.")
            x.fillna(x.mean(), inplace=True)  # پر کردن مقادیر گمشده با میانگین ستون‌ها

        # تقسیم داده‌ها به آموزش و آزمون
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, random_state=42
        )
        logging.info("داده‌ها به صورت موفقیت‌آمیز به دو بخش آموزش و آزمون تقسیم شدند.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"خطا در بارگذاری یا پیش‌پردازش داده‌ها: {e}")
        raise


def train_and_evaluate_svm(x_train, y_train, x_test, y_test):
    """
    آموزش و ارزیابی مدل SVM.

    این تابع یک مدل SVM را با استفاده از RandomizedSearchCV آموزش می‌دهد و پارامترهای بهینه را پیدا می‌کند.
    سپس مدل بهینه شده را روی داده‌های آزمون ارزیابی می‌کند و نتایج را نمایش می‌دهد.

    Args:
        x_train (pd.DataFrame): داده‌های ویژگی‌ها برای آموزش.
        y_train (pd.Series): برچسب‌های هدف برای آموزش.
        x_test (pd.DataFrame): داده‌های ویژگی‌ها برای آزمون.
        y_test (pd.Series): برچسب‌های هدف برای آزمون.

    Returns:
        y_test (pd.Series): برچسب‌های واقعی داده‌های آزمون.
        y_pred (np.ndarray): برچسب‌های پیش‌بینی‌شده توسط مدل.
        y_proba (np.ndarray): احتمالات پیش‌بینی‌شده برای کلاس مثبت.
        best_model (Pipeline): مدل بهینه‌شده.
    """
    try:
        # ایجاد Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # استانداردسازی داده‌ها
            ('svc', SVC(probability=True))  # probability=True برای محاسبه ROC-AUC
        ])

        # تعریف توزیع پارامترها
        param_dist = {
            'svc__C': np.logspace(-3, 3, 7),  # مقادیر مختلف برای پارامتر C
            'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # انواع هسته‌ها
            'svc__gamma': ['scale', 'auto'],  # مقادیر مختلف برای پارامتر Gamma
            'svc__degree': [2, 3, 4],  # درجه برای هسته Polynomial
            'svc__coef0': [0.0, 1.0, 2.0]  # ضریب برای هسته‌های Polynomial و Sigmoid
        }

        # جستجوی تصادفی برای تنظیم پارامترها
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=100,  # تعداد تکرارها برای جستجوی تصادفی
            cv=5,  # استفاده از 5-Fold Cross Validation
            scoring='f1',  # استفاده از F1-Score به عنوان معیار ارزیابی
            random_state=42,
            n_jobs=-1  # استفاده از تمام هسته‌های CPU
        )
        random_search.fit(x_train, y_train)
        logging.info("پارامترهای بهینه با RandomizedSearchCV پیدا شدند.")

        # نمایش نتایج
        print("Best Parameters:", random_search.best_params_)
        print("Best Cross-Validation F1-Score:", random_search.best_score_)

        # ارزیابی مدل روی داده‌های آزمون
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(x_test)
        y_proba = best_model.predict_proba(x_test)[:, 1]  # احتمالات برای ROC-AUC
        test_accuracy = accuracy_score(y_test, y_pred)
        print("\nTest Accuracy:", test_accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return y_test, y_pred, y_proba, best_model
    except Exception as e:
        logging.error(f"خطا در آموزش یا ارزیابی مدل SVM: {e}")
        raise


def visualize_results(y_test, y_pred, y_proba, best_model, x, y):
    """
    تجسم نتایج مدل.

    این تابع نتایج مدل را با استفاده از نمودارهای مختلف تجسم می‌کند:
    1. Confusion Matrix: برای نمایش نتایج پیش‌بینی.
    2. ROC Curve: برای نمایش عملکرد مدل بر اساس ROC-AUC.
    3. Precision-Recall Curve: برای نمایش تعادل بین Precision و Recall.
    4. Cross-Validation Accuracy: برای نمایش دقت در هر فولد.

    Args:
        y_test (pd.Series): برچسب‌های واقعی داده‌های آزمون.
        y_pred (np.ndarray): برچسب‌های پیش‌بینی‌شده توسط مدل.
        y_proba (np.ndarray): احتمالات پیش‌بینی‌شده برای کلاس مثبت.
        best_model (Pipeline): مدل بهینه‌شده.
        x (pd.DataFrame): داده‌های ویژگی‌ها.
        y (pd.Series): برچسب‌های هدف.
    """
    try:
        # نمودار Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # نمودار ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

        # نمودار Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()

        # نمودار Cross-Validation Accuracy
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, x, y, cv=kf, scoring='accuracy')
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 6), cv_scores, marker='o', linestyle='--')
        plt.axhline(y=np.mean(cv_scores), color='r', linestyle='-', label=f'Mean Accuracy: {np.mean(cv_scores):.3f}')
        plt.title('Cross-Validation Accuracy for Each Fold')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logging.error(f"خطا در تجسم نتایج: {e}")
        raise


def main():
    """
    تابع اصلی برای اجرای کد.

    این تابع مراحل زیر را انجام می‌دهد:
    1. بارگذاری و پیش‌پردازش داده‌ها.
    2. آموزش و ارزیابی مدل SVM.
    3. تجسم نتایج مدل.
    """
    try:
        # بارگذاری و پیش‌پردازش داده‌ها
        x_train, x_test, y_train, y_test = load_and_preprocess_data()

        # آموزش و ارزیابی مدل SVM
        y_test, y_pred, y_proba, best_model = train_and_evaluate_svm(x_train, y_train, x_test, y_test)

        # بارگذاری داده‌های کامل برای تجسم نتایج
        x, y = datasets.load_breast_cancer(as_frame=True, return_X_y=True)

        # تجسم نتایج
        visualize_results(y_test, y_pred, y_proba, best_model, x, y)
    except Exception as e:
        logging.error(f"خطا در اجرای برنامه: {e}")


if __name__ == "__main__":
    main()