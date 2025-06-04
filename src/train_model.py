import joblib  # For saving/loading models
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series, model_type: str, random_state: int = 42):
    """
    Trains a specified classification model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        model_type (str): Type of model to train ('LogisticRegression' or 'RandomForestClassifier').
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        model: A trained scikit-learn classifier model.

    Raises:
        ValueError: If an unsupported model_type is provided.
    """
    if model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=random_state, solver='liblinear')
    elif model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'LogisticRegression' or 'RandomForestClassifier'.")

    print(f"Training {model_type}...")
    model.fit(X_train, y_train)
    print(f"{model_type} training complete.")
    return model

if __name__ == '__main__':
    print("Running train_model.py as a script for testing...")
    
    # Create dummy data for testing
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_s = pd.Series(y, name='target')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.0, random_state=42, stratify=y_s) # No test set for training test

    print("\n--- Testing Logistic Regression ---")
    lr_model = train_classifier(X_train, y_train, 'LogisticRegression')
    try:
        # Test if the model is fitted
        coefs = lr_model.coef_
        print(f"Logistic Regression model fitted. First 5 coefficients: {coefs[0][:5]}")
    except NotFittedError:
        print("Logistic Regression model was not fitted.")

    print("\n--- Testing Random Forest Classifier ---")
    rf_model = train_classifier(X_train, y_train, 'RandomForestClassifier')
    try:
        # Test if the model is fitted
        feature_importances = rf_model.feature_importances_
        print(f"Random Forest model fitted. Sum of feature importances: {feature_importances.sum():.2f}")
    except NotFittedError:
        print("Random Forest model was not fitted.")
    
    # Example of saving a trained model
    joblib.dump(lr_model, 'dummy_lr_model.joblib')
    print("\nDummy Logistic Regression model saved to 'dummy_lr_model.joblib'")
    
    # Example of loading a trained model
    loaded_lr_model = joblib.load('dummy_lr_model.joblib')
    print("Dummy Logistic Regression model loaded from 'dummy_lr_model.joblib'")
    import os
    os.remove('dummy_lr_model.joblib') # Clean up the dummy file
    print("Cleaned up dummy model file.")