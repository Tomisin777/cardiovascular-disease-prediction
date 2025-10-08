"""
Random Forest model for cardiovascular disease prediction.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        RandomForestClassifier: Trained model
    """
    rf_model = RandomForestClassifier(random_state=config.RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate Random Forest model.
    
    Args:
        model: Trained random forest model
        X_test: Testing features
        y_test: Testing labels
    
    Returns:
        tuple: (predictions, accuracy)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy