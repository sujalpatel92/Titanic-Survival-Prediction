import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator  # For type hinting models
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_classifier(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"):
    """
    Evaluates a trained classification model and prints key metrics along with visualizations.

    Args:
        model (BaseEstimator): A trained scikit-learn classifier model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True target values for the test set.
        model_name (str): Name of the model for display purposes.
    """
    if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
        print(f"Error: {model_name} does not seem to be a valid scikit-learn classifier (missing predict or predict_proba).")
        return

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Evaluation ---")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    # Plot ROC curve for the single model
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    print("Running evaluate_model.py as a script for testing...")
    
    # Create dummy data and a dummy model for testing
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_s = pd.Series(y, name='target')

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.3, random_state=42, stratify=y_s)

    # Train a dummy Logistic Regression model
    dummy_model = LogisticRegression(random_state=42, solver='liblinear')
    dummy_model.fit(X_train, y_train)

    print("\n--- Evaluating Dummy Logistic Regression Model ---")
    evaluate_classifier(dummy_model, X_test, y_test, 'Dummy Logistic Regression')

    # You could also test with a Random Forest model here if needed
    # from sklearn.ensemble import RandomForestClassifier
    # dummy_rf_model = RandomForestClassifier(random_state=42)
    # dummy_rf_model.fit(X_train, y_train)
    # evaluate_classifier(dummy_rf_model, X_test, y_test, 'Dummy Random Forest')