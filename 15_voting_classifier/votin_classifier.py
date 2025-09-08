import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import VotingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Load dataset
x, y = load_breast_cancer(as_frame=True, return_X_y=True)
# Plot class distribution

plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette='Set2', hue=y)
plt.title('Class Distribution (Benign vs Malignant)', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['Benign', 'Malignant'], fontsize=12)
plt.show()

# Calculate correlation matrix
correlation_matrix = x.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.show()


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(x_train)

# فقط داده‌های نرمال را نگه دارید
mask = outliers == 1
x_train = x_train[mask]
y_train = y_train[mask]

# ==================== Step 1: Optimize Logistic Regression ====================
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear'],  # Solver compatible with both L1 and L2 penalties
    'max_iter': [500, 1000]
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
best_logistic_model = grid_search.fit(x_train, y_train).best_estimator_

# ==================== Step 2: Define Models ====================
models = [
    ("logistic", best_logistic_model),  # Optimized Logistic Regression
    ("svm", SVC(probability=True)),  # SVM with probability estimates for Soft Voting
    ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest
]

# ==================== Step 3: Create and Train Voting Classifier ====================
voting_clf = VotingClassifier(estimators=models, voting='soft')  # Use 'soft' voting for probability-based decisions
voting_clf.fit(x_train, y_train)

# ==================== Step 4: Evaluate Voting Classifier ====================
# Predict on test set
voting_pred = voting_clf.predict(x_test)

# Calculate accuracy
voting_accuracy = round(accuracy_score(voting_pred, y_test), 3)
print(f'Voting Classifier Accuracy: {voting_accuracy}')

# Calculate recall
voting_recall = round(recall_score(y_test, voting_pred), 3)
print(f'Voting Classifier Recall: {voting_recall}')

# Calculate precision
voting_precision = round(precision_score(y_test, voting_pred), 3)
print(f'Voting Classifier Precision: {voting_precision}')

# Calculate F1-score
voting_f1 = round(f1_score(y_test, voting_pred), 3)
print(f'Voting Classifier F1-Score: {voting_f1}')


# Cross-validation for more robust evaluation
cv_scores = cross_val_score(voting_clf, x, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy (Mean): {round(cv_scores.mean(), 3)}')


# Create a DataFrame for performance metrics
performance_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1-Score'],
    'Score': [voting_accuracy, voting_recall, voting_precision, voting_f1]
})

# Plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Score',hue='Metric', data=performance_metrics, palette='viridis')
plt.title('Voting Classifier Performance Metrics', fontsize=16)
plt.ylim(0, 1)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Metric', fontsize=14)
plt.show()


# Get predicted probabilities for the positive class
y_prob = voting_clf.predict_proba(x_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.show()

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, voting_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Benign', 'Predicted Malignant'],
            yticklabels=['Actual Benign', 'Actual Malignant'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.show()

# Get feature importances from Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)
feature_importances = rf_model.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature',hue='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.show()

# Plot cross-validation scores
plt.figure(figsize=(8, 6))
sns.barplot(x=range(1, 6), y=cv_scores,hue=range(1,6), palette='magma')
plt.title('Cross-Validation Accuracy Scores', fontsize=16)
plt.xlabel('Fold', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)
plt.show()

# Apply PCA to reduce data to 2 dimensions
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# Apply PCA to reduce data to 2 dimensions
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(x)

# Plot data points in 2D space
plt.figure(figsize=(12, 10))
scatter_plot = sns.scatterplot(
    x=x_pca[:, 0],
    y=x_pca[:, 1],
    hue=y,  # Assign the class labels to `hue`
    palette='coolwarm',  # Use a proper color palette
    s=150,
    edgecolor='k',
    alpha=0.8,
    legend=True  # Ensure legend is displayed
)

# Add title and labels
plt.title('PCA Visualization of Data (2D)', fontsize=18, fontweight='bold')
plt.xlabel('Principal Component 1 (Explains Most Variance)', fontsize=16)
plt.ylabel('Principal Component 2 (Second Most Variance)', fontsize=16)

# Customize legend
legend_labels = ['Benign (Non-Cancerous)', 'Malignant (Cancerous)']
plt.legend(title='Class Labels', labels=legend_labels, fontsize=12, title_fontsize=14)

# Add grid for better readability
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# ==================== Step 5: Test with a Manual Data Point ====================
# Example manual data point (replace with actual feature values)
# Suppose the dataset has 30 features (as in Breast Cancer dataset)

# Create manual data as a DataFrame with feature names
feature_names = x.columns  # Get feature names from the original dataset
manual_data = pd.DataFrame([
    [
        11.2, 29.37, 70.67, 386, 0.07449, 0.03558, 0, 0,
        0.106, 0.05502, 0.3141, 3.896, 2.041, 22.81, 0.007594,
        0.008878, 0, 0, 0.01989, 0.001773, 11.92, 38.3,
        75.19, 439.6, 0.09267, 0.05494, 0, 0, 0.1566, 0.05905
    ]
], columns=feature_names)

# Scale the manual data using the same scaler used for training
manual_data_scaled = scaler.transform(manual_data)

# Predict using the Voting Classifier
manual_prediction = voting_clf.predict(manual_data_scaled)

# Interpret the result
if manual_prediction[0] == 1:
    print("The model predicts that the sample is Malignant (Cancerous).")
else:
    print("The model predicts that the sample is Benign (Non-Cancerous).")


# ==================== Step 5: Decision Tree Ensemble (Optional) ====================
# Create a list of Decision Tree models with varying depths
decision_tree_models = [DecisionTreeClassifier(max_depth=i, random_state=42) for i in range(2, 22)]

# Create a Voting Classifier with Decision Trees
tree_voting_clf = VotingClassifier(estimators=[(f"model_{i}", model) for i, model in enumerate(decision_tree_models)])
tree_voting_clf.fit(x_train, y_train)

# Predict and evaluate
tree_voting_pred = tree_voting_clf.predict(x_test)
tree_voting_accuracy = round(accuracy_score(tree_voting_pred, y_test), 3)
print(f'Decision Tree Voting Classifier Accuracy: {tree_voting_accuracy}')
