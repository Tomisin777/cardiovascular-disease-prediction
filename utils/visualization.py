"""
Visualization utilities for model comparison, confusion matrices, and ROC curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)


def plot_model_comparison(accuracies_dict, save_path=config.MODEL_COMPARISON_PLOT):
    """
    Create bar plot comparing model accuracies.
    
    Args:
        accuracies_dict (dict): Dictionary mapping model names to accuracy scores
        save_path (str): Path to save the plot
    """
    ensure_output_dir()
    
    models = list(accuracies_dict.keys())
    accuracies = list(accuracies_dict.values())
    
    plt.figure(figsize=config.FIGURE_SIZE_LARGE, dpi=config.PLOT_DPI)
    plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0.7, 1.0])
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Model comparison plot saved to {save_path}")


def plot_confusion_matrix(cm, title='Confusion Matrix', 
                         save_path=config.CONFUSION_MATRIX_PLOT,
                         cmap='Blues'):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        title (str): Plot title
        save_path (str): Path to save the plot
        cmap (str): Color map for heatmap
    """
    ensure_output_dir()
    
    plt.figure(figsize=config.FIGURE_SIZE_MEDIUM, dpi=config.PLOT_DPI)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_optimized_confusion_matrix(cm, threshold, 
                                   save_path=config.OPTIMIZED_CM_PLOT):
    """
    Plot confusion matrix for optimized threshold model.
    
    Args:
        cm: Confusion matrix
        threshold (float): The threshold used
        save_path (str): Path to save the plot
    """
    title = f'Optimized Model (Threshold={threshold})'
    plot_confusion_matrix(cm, title=title, save_path=save_path, cmap='Greens')


def plot_roc_curve(fpr, tpr, roc_auc, model_name='Model', 
                   save_path=None):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the ROC curve
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    ensure_output_dir()
    
    if save_path is None:
        save_path = config.OUTPUT_DIR + 'roc_curve.png'
    
    plt.figure(figsize=config.FIGURE_SIZE_MEDIUM, dpi=config.PLOT_DPI)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"ROC curve saved to {save_path}")


def plot_multiple_roc_curves(models_metrics, save_path=None):
    """
    Plot multiple ROC curves on the same plot for comparison.
    
    Args:
        models_metrics (dict): Dictionary mapping model names to their metrics
                              (must contain 'fpr', 'tpr', 'roc_auc')
        save_path (str): Path to save the plot
    """
    ensure_output_dir()
    
    if save_path is None:
        save_path = config.OUTPUT_DIR + 'roc_curves_comparison.png'
    
    plt.figure(figsize=config.FIGURE_SIZE_LARGE, dpi=config.PLOT_DPI)
    
    colors = ['darkorange', 'green', 'red', 'purple', 'brown']
    
    for idx, (model_name, metrics) in enumerate(models_metrics.items()):
        if 'fpr' in metrics and 'tpr' in metrics and 'roc_auc' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'], 
                    color=colors[idx % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {metrics["roc_auc"]:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"ROC curves comparison saved to {save_path}")


def plot_metrics_comparison(models_metrics, save_path=None):
    """
    Create a bar chart comparing multiple metrics across models.
    
    Args:
        models_metrics (dict): Dictionary mapping model names to their metrics
        save_path (str): Path to save the plot
    """
    ensure_output_dir()
    
    if save_path is None:
        save_path = config.OUTPUT_DIR + 'metrics_comparison.png'
    
    models = list(models_metrics.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Prepare data
    data = {metric: [] for metric in metrics_names}
    for model in models:
        for metric in metrics_names:
            data[metric].append(models_metrics[model].get(metric, 0))
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_LARGE, dpi=config.PLOT_DPI)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, metric in enumerate(metrics_names):
        offset = width * (idx - 1.5)
        ax.bar(x + offset, data[metric], width, label=metric.capitalize(), 
               color=colors[idx])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Metrics comparison saved to {save_path}")