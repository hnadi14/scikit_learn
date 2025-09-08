import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load dataset
data = load_breast_cancer(as_frame=True)
x, y = data.data, data.target

# Step 2: Handle Imbalanced Data (if needed)
# Check if the dataset is imbalanced
print("Class distribution:\n", y.value_counts())

# Plot class distribution before SMOTE
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# If imbalanced, use SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# x, y = smote.fit_resample(x, y)

# Plot class distribution after SMOTE
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Step 3: Feature Selection
# Select top features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=20)  # Select top 20 features
x_selected = selector.fit_transform(x, y)

# Print selected features
selected_features = x.columns[selector.get_support()]
print("\nSelected Features:", selected_features.tolist())

# Plot feature importance
feature_scores = selector.scores_[selector.get_support()]
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_scores, y=selected_features)
plt.title('Feature Importance (ANOVA F-test)')
plt.xlabel('F-score')
plt.ylabel('Features')
plt.show()

# Step 4: Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
x_pca = pca.fit_transform(x_selected)

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Explained Variance Ratio vs Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

print(f"Reduced dimensions from {x_selected.shape[1]} to {x_pca.shape[1]}")

# Step 5: Split data into training and testing sets with shuffle
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.33, shuffle=True, random_state=42)

# Step 6: Create a Pipeline for preprocessing and modeling
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('knn', KNeighborsClassifier())  # KNN model
])

# Step 7: Use GridSearchCV with advanced hyperparameter tuning
param_grid = {
    'knn__n_neighbors': np.arange(1, 31),  # Search for k in range 1 to 30
    'knn__weights': ['uniform', 'distance'],  # Uniform or distance-based weighting
    'knn__metric': ['euclidean', 'manhattan']  # Distance metrics
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Get the best parameters and train the final model
best_params = grid_search.best_params_
print(f"\nBest Parameters: {best_params}")

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(
        n_neighbors=best_params['knn__n_neighbors'],
        weights=best_params['knn__weights'],
        metric=best_params['knn__metric']
    ))
])
final_pipeline.fit(x_train, y_train)

# Step 8: Make predictions
y_pred = final_pipeline.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 9: Compare with other models (Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {accuracy_rf:.4f}")

# Step 10: Plot Model Comparison
models = ['KNN', 'Random Forest']
accuracies = [accuracy, accuracy_rf]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.title('Model Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Step 11: ROC-AUC Curve
y_prob_knn = final_pipeline.predict_proba(x_test)[:, 1]
y_prob_rf = rf.predict_proba(x_test)[:, 1]

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})', color='blue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.title('ROC-AUC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Step 12: Validation Curve for KNN
k_values = np.arange(1, 31)
mean_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    mean_scores.append(scores.mean())

plt.figure(figsize=(8, 5))
plt.plot(k_values, mean_scores, marker='o')
plt.title('Validation Curve for KNN')
plt.xlabel('k')
plt.ylabel('Mean Accuracy')
plt.grid()
plt.show()