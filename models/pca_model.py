"""
PCA dimensionality reduction with Logistic Regression.
"""

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def apply_pca(X_train, X_test, n_components=config.PCA_COMPONENTS):
    """
    Apply PCA dimensionality reduction.
    
    Args:
        X_train: Training features
        X_test: Testing features
        n_components (int): Number of principal components
    
    Returns:
        tuple: (X_train_pca, X_test_pca, pca_model)
    """
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def train_pca_logistic(X_train_pca, y_train):
    """
    Train Logistic Regression on PCA-transformed data.
    
    Args:
        X_train_pca: PCA-transformed training features
        y_train: Training labels
    
    Returns:
        LogisticRegression: Trained model
    """
    model_pca = LogisticRegression(
        random_state=config.RANDOM_STATE,
        max_iter=config.LOGISTIC_MAX_ITER
    )
    model_pca.fit(X_train_pca, y_train)
    return model_pca


def evaluate_pca_model(model, X_test_pca, y_test):
    """
    Evaluate PCA + Logistic Regression model.
    
    Args:
        model: Trained PCA logistic model
        X_test_pca: PCA-transformed testing features
        y_test: Testing labels
    
    Returns:
        tuple: (predictions, accuracy)
    """
    y_pred_pca = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred_pca)
    return y_pred_pca, accuracy