"""
Main script for cardiovascular disease prediction.
Runs the complete pipeline: data loading, model training, evaluation, and visualization.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config
from data.load_data import prepare_data
from models.logistic_model import (
    train_logistic_regression, 
    evaluate_logistic_regression,
    get_probabilities
)
from models.random_forest_model import train_random_forest, evaluate_random_forest
from models.pca_model import apply_pca, train_pca_logistic, evaluate_pca_model
from utils.evaluation import (
    test_thresholds, 
    apply_optimal_threshold,
    calculate_comprehensive_metrics,
    print_metrics_report
)
from utils.visualization import (
    plot_model_comparison,
    plot_confusion_matrix,
    plot_optimized_confusion_matrix,
    plot_roc_curve,
    plot_multiple_roc_curves,
    plot_metrics_comparison
)
from sklearn.metrics import confusion_matrix


def main():
    """
    Main execution function for the cardiovascular disease prediction pipeline.
    """
    print("="*60)
    print("CARDIOVASCULAR DISEASE PREDICTION MODEL")
    print("="*60)
    
    # Set numpy random seed for reproducibility
    np.random.seed(config.RANDOM_STATE)
    
    # Step 1: Prepare data
    print("\n[STEP 1] DATA PREPARATION")
    print("-"*60)
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Step 2: Train and evaluate Logistic Regression
    print("\n[STEP 2] LOGISTIC REGRESSION")
    print("-"*60)
    lr_model = train_logistic_regression(X_train, y_train)
    y_pred_lr, lr_accuracy = evaluate_logistic_regression(lr_model, X_test, y_test)
    lr_proba = get_probabilities(lr_model, X_test)
    lr_metrics = calculate_comprehensive_metrics(y_test, y_pred_lr, lr_proba)
    print_metrics_report(lr_metrics, "Logistic Regression")
    
    # Step 3: Train and evaluate PCA + Logistic Regression
    print("\n[STEP 3] PCA + LOGISTIC REGRESSION")
    print("-"*60)
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test)
    pca_model = train_pca_logistic(X_train_pca, y_train)
    y_pred_pca, pca_accuracy = evaluate_pca_model(pca_model, X_test_pca, y_test)
    pca_proba = pca_model.predict_proba(X_test_pca)[:, 1]
    pca_metrics = calculate_comprehensive_metrics(y_test, y_pred_pca, pca_proba)
    print_metrics_report(pca_metrics, "PCA + Logistic Regression")
    
    # Step 4: Train and evaluate Random Forest
    print("\n[STEP 4] RANDOM FOREST")
    print("-"*60)
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf, rf_accuracy = evaluate_random_forest(rf_model, X_test, y_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = calculate_comprehensive_metrics(y_test, y_pred_rf, rf_proba)
    print_metrics_report(rf_metrics, "Random Forest")
    
    # Step 5: Compare models - Basic accuracy
    print("\n[STEP 5] MODEL COMPARISON - ACCURACY")
    print("-"*60)
    accuracies = {
        'Logistic Regression': lr_accuracy,
        'PCA + Logistic': pca_accuracy,
        'Random Forest': rf_accuracy
    }
    print("\nModel Accuracies:")
    for model_name, acc in accuracies.items():
        print(f"  {model_name}: {acc:.4f}")
    
    # Step 6: Visualize model comparison (accuracy)
    print("\n[STEP 6] VISUALIZATION - Model Comparison (Accuracy)")
    print("-"*60)
    plot_model_comparison(accuracies)
    
    # Step 7: Visualize comprehensive metrics comparison
    print("\n[STEP 7] VISUALIZATION - Comprehensive Metrics")
    print("-"*60)
    all_models_metrics = {
        'Logistic Regression': lr_metrics,
        'PCA + Logistic': pca_metrics,
        'Random Forest': rf_metrics
    }
    plot_metrics_comparison(all_models_metrics)
    
    # Step 8: ROC Curves for all models
    print("\n[STEP 8] VISUALIZATION - ROC Curves")
    print("-"*60)
    plot_multiple_roc_curves(all_models_metrics)
    
    # Step 9: Confusion matrix for base logistic regression
    print("\n[STEP 9] CONFUSION MATRIX - Base Logistic Regression")
    print("-"*60)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plot_confusion_matrix(cm_lr, title='Confusion Matrix - Logistic Regression')
    
    # Step 10: Threshold optimization
    print("\n[STEP 10] THRESHOLD OPTIMIZATION")
    print("-"*60)
    disease_proba = get_probabilities(lr_model, X_test)
    threshold_results = test_thresholds(y_test, disease_proba)
    
    # Step 11: Apply optimal threshold
    print("\n[STEP 11] FINAL MODEL WITH OPTIMAL THRESHOLD")
    print("-"*60)
    final_results = apply_optimal_threshold(y_test, disease_proba)
    
    # Step 12: Visualize optimized model
    print("\n[STEP 12] VISUALIZATION - Optimized Model")
    print("-"*60)
    plot_optimized_confusion_matrix(
        final_results['confusion_matrix'],
        config.OPTIMAL_THRESHOLD
    )
    
    # Step 13: ROC curve for optimized model
    print("\n[STEP 13] VISUALIZATION - Optimized Model ROC")
    print("-"*60)
    plot_roc_curve(
        final_results['fpr'], 
        final_results['tpr'], 
        final_results['roc_auc'],
        model_name=f'Optimized (Threshold={config.OPTIMAL_THRESHOLD})',
        save_path=config.OUTPUT_DIR + 'roc_curve_optimized.png'
    )
    
    # Final Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll outputs saved to '{config.OUTPUT_DIR}' directory")
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print("\nBest Base Model (by accuracy):")
    best_model = max(accuracies, key=accuracies.get)
    print(f"  Model: {best_model}")
    print(f"  Accuracy: {accuracies[best_model]:.4f}")
    
    print("\nOptimized Model Performance:")
    print(f"  Threshold: {config.OPTIMAL_THRESHOLD}")
    print(f"  Accuracy: {final_results['accuracy']:.4f}")
    print(f"  Precision: {final_results['precision']:.4f}")
    print(f"  Sensitivity (Recall): {final_results['sensitivity']:.4f}")
    print(f"  Specificity: {final_results['specificity']:.4f}")
    print(f"  F1-Score: {final_results['f1_score']:.4f}")
    print(f"  ROC-AUC: {final_results['roc_auc']:.4f}")
    
    print("\n" + "="*60)
    print("Key Insight:")
    print(f"The optimized model achieves {final_results['sensitivity']:.1%} sensitivity")
    print(f"(catches {final_results['sensitivity']:.1%} of disease cases)")
    print(f"at the cost of {final_results['specificity']:.1%} specificity.")
    print("="*60)


if __name__ == "__main__":
    main()