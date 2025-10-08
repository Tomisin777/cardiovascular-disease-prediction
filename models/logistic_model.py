"""
Logistic Regression model for cardiovascular disease prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(
        random_state=config.RANDOM_STATE,
        max_iter=config.LOGISTIC_MAX_ITER
    )
    model.fit(X_train, y_train)
    return model


def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate Logistic Regression model.
    
    Args:
        model: Trained logistic regression model
        X_test: Testing features
        y_test: Testing labels
    
    Returns:
        tuple: (predictions, accuracy)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy


def get_probabilities(model, X_test):
    """
    Get prediction probabilities from the model.
    
    Args:
        model: Trained logistic regression model
        X_test: Testing features
    
    Returns:
        np.array: Probability of disease (positive class)
    """
    y_prob = model.predict_proba(X_test)
    disease_proba = y_prob[:, 1]
    return disease_proba