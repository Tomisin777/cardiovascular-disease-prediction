"""
Configuration file for cardiovascular disease prediction model.
Contains all hyperparameters, paths, and settings.
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# Data settings
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
COLUMN_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Train/test split
TEST_SIZE = 0.2

# Model hyperparameters
LOGISTIC_MAX_ITER = 1000
PCA_COMPONENTS = 5

# Threshold settings
DEFAULT_THRESHOLD = 0.5
OPTIMAL_THRESHOLD = 0.15
THRESHOLDS_TO_TEST = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]

# Visualization settings
FIGURE_SIZE_LARGE = (10, 6)
FIGURE_SIZE_MEDIUM = (8, 6)
PLOT_DPI = 100

# Output paths
OUTPUT_DIR = 'outputs/'
MODEL_COMPARISON_PLOT = OUTPUT_DIR + 'model_comparison.png'
CONFUSION_MATRIX_PLOT = OUTPUT_DIR + 'confusion_matrix.png'
OPTIMIZED_CM_PLOT = OUTPUT_DIR + 'confusion_matrix_optimized.png'