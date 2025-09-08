import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import export_text
from sklearn.datasets import load_breast_cancer
import logging

# تنظیمات لاگ‌گیری
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data():
    """
    Load the Breast Cancer dataset and split it into training and testing sets.
    Returns:
        x_train, x_test, y_train, y_test
    """
    logging.info("Loading and splitting the dataset...")
    x, y = load_breast_cancer(as_frame=True, return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


def train_adaboost(x_train, y_train, n_estimators=1000, random_state=42):
    """
    Train an AdaBoost classifier.
    Args:
        x_train: Training features
        y_train: Training labels
        n_estimators: Number of weak learners
        random_state: Random seed for reproducibility
    Returns:
        Trained AdaBoost classifier
    """
    logging.info(f"Training AdaBoost with {n_estimators} weak learners...")
    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    ada.fit(x_train, y_train)
    return ada


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on the test set and calculate accuracy.
    Args:
        model: Trained model
        x_test: Test features
        y_test: Test labels
    Returns:
        Accuracy score
    """
    logging.info("Evaluating the model...")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
    return accuracy


def display_weak_learners(model, feature_names, num_learners=5):
    """
    Display the structure of the first few weak learners in the AdaBoost model.
    Args:
        model: Trained AdaBoost classifier
        feature_names: List of feature names
        num_learners: Number of weak learners to display
    """
    logging.info(f"Displaying the first {num_learners} weak learners...")
    weak_learners = model.estimators_
    weights = model.estimator_weights_  # Weights of weak learners
    for i, (estimator, weight) in enumerate(zip(weak_learners[:num_learners], weights[:num_learners])):
        logging.info(f"\nWeak Learner {i + 1} (Weight: {weight:.4f}):")
        tree_rules = export_text(estimator, feature_names=feature_names)
        logging.info(tree_rules)


def main():
    """
    Main function to execute the entire pipeline.
    """
    # Load data
    x_train, x_test, y_train, y_test = load_data()

    # Train AdaBoost model
    ada_model = train_adaboost(x_train, y_train)

    # Evaluate model
    evaluate_model(ada_model, x_test, y_test)

    # Display weak learners
    display_weak_learners(ada_model, list(x_train.columns))


if __name__ == "__main__":
    main()