"""
Model evaluation utilities including comprehensive metrics and threshold optimization.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def calculate_sensitivity(cm):
    """
    Calculate sensitivity (recall for positive class).
    
    Sensitivity = True Positives / (True Positives + False Negatives)
    
    Args:
        cm: Confusion matrix
    
    Returns:
        float: Sensitivity score
    """
    return cm[1, 1] / (cm[1, 0] + cm[1, 1])


def calculate_specificity(cm):
    """
    Calculate specificity (recall for negative class).
    
    Specificity = True Negatives / (True Negatives + False Positives)
    
    Args:
        cm: Confusion matrix
    
    Returns:
        float: Specificity score
    """
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])


def calculate_comprehensive_metrics(y_test, y_pred, y_proba=None):
    """
    Calculate comprehensive evaluation metrics for a classification model.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),  # Same as sensitivity
        'sensitivity': calculate_sensitivity(cm),
        'specificity': calculate_specificity(cm),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': cm,
        'false_negatives': cm[1, 0],
        'false_positives': cm[0, 1],
        'true_negatives': cm[0, 0],
        'true_positives': cm[1, 1]
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['roc_thresholds'] = thresholds
    
    return metrics


def print_metrics_report(metrics, model_name="Model"):
    """
    Print a formatted report of all metrics.
    
    Args:
        metrics (dict): Dictionary of metrics from calculate_comprehensive_metrics
        model_name (str): Name of the model for the report
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - COMPREHENSIVE METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity: {metrics['recall']:.4f}")
    print(f"Specificity:     {metrics['specificity']:.4f}")
    print(f"F1-Score:        {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               No    Yes")
    print(f"Actual No  [{metrics['true_negatives']:4d}  {metrics['false_positives']:4d}]")
    print(f"       Yes [{metrics['false_negatives']:4d}  {metrics['true_positives']:4d}]")
    print(f"\nFalse Negatives (Missed Disease): {metrics['false_negatives']}")
    print(f"False Positives (False Alarms):   {metrics['false_positives']}")
    print(f"{'='*60}")


def test_thresholds(y_test, disease_proba, thresholds=config.THRESHOLDS_TO_TEST):
    """
    Test multiple classification thresholds and evaluate metrics.
    
    Args:
        y_test: True labels
        disease_proba: Predicted probabilities
        thresholds (list): List of thresholds to test
    
    Returns:
        dict: Results for each threshold
    """
    results = {}
    
    print(f"\n{'='*60}")
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    
    for threshold in thresholds:
        y_pred_new = (disease_proba >= threshold).astype(int)
        metrics = calculate_comprehensive_metrics(y_test, y_pred_new, disease_proba)
        
        results[threshold] = metrics
        
        print(f"\nThreshold = {threshold}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  False Neg:   {metrics['false_negatives']}")
        print(f"  False Pos:   {metrics['false_positives']}")
    
    return results


def apply_optimal_threshold(y_test, disease_proba, threshold=config.OPTIMAL_THRESHOLD):
    """
    Apply optimal threshold and calculate final metrics.
    
    Args:
        y_test: True labels
        disease_proba: Predicted probabilities
        threshold (float): Optimal threshold value
    
    Returns:
        dict: Final comprehensive metrics
    """
    y_pred_optimized = (disease_proba >= threshold).astype(int)
    metrics = calculate_comprehensive_metrics(y_test, y_pred_optimized, disease_proba)
    
    print(f"\n{'='*60}")
    print(f"FINAL MODEL (Optimized Threshold = {threshold})")
    print(f"{'='*60}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity:     {metrics['specificity']:.4f}")
    print(f"F1-Score:        {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nFalse Negatives (Missed): {metrics['false_negatives']}")
    print(f"False Positives (False Alarms): {metrics['false_positives']}")
    print(f"{'='*60}")
    
    # Add predictions to metrics
    metrics['predictions'] = y_pred_optimized
    
    return metrics