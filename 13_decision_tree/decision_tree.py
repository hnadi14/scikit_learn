import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

x , y=load_breast_cancer(return_X_y=True, as_frame=True)
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.33, random_state=42)

dtc=DecisionTreeClassifier(max_depth=3 , class_weight={0:0.4,1:0.6})
dtc.fit(x_train, y_train)

plt.figure(figsize=(20, 10))  # تنظیم اندازه نمودار
plot = plot_tree(
    dtc,
    feature_names=x.columns,  # نام ویژگی‌ها
    class_names=["Malignant", "Benign"],  # نام کلاس‌ها
    filled=True,
    rounded=True
)

# Save the tree image to a file
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")  # Save as PNG with high resolution
plt.show()

y_pred = dtc.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
prob=dtc.predict_proba(x_test)
print(prob)

print(export_text(dtc, feature_names=x.columns, class_names=["Malignant", "Benign"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)  # Calculate confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Malignant", "Benign"])
disp.plot(cmap=plt.cm.Blues)  # Use a blue colormap for better visualization
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")  # Save confusion matrix as image
plt.show()  # Show the confusion matrix

# Precision and Recall for each class
precision_per_class = metrics.precision_score(y_test, y_pred, average=None)  # Precision for each class
recall_per_class = metrics.recall_score(y_test, y_pred, average=None)  # Recall for each class

# Print Precision and Recall
print(f"Precision (Malignant): {precision_per_class[0]:.2f}")
print(f"Precision (Benign): {precision_per_class[1]:.2f}")
print(f"Recall (Malignant): {recall_per_class[0]:.2f}")
print(f"Recall (Benign): {recall_per_class[1]:.2f}")