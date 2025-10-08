# Cardiovascular Disease Prediction

This project explores how machine learning can be applied to predict cardiovascular disease using patient health data. Built as a personal learning exercise, it combines data preprocessing, feature engineering, and model comparison to translate classroom concepts into a real-world biomedical challenge. Beyond the code, it reflects my curiosity about how data-driven approaches can support early diagnosis and personalized care in cardiovascular health.

## Project Overview

Cardiovascular disease is one of the leading causes of death globally. Early detection and risk assessment are crucial for prevention and treatment. This project builds predictive models using the UCI Heart Disease dataset to classify patients as having or not having heart disease based on clinical features.

**Key Features:**
- Multiple ML models (Logistic Regression, Random Forest, PCA)
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ROC curve analysis and visualization
- Threshold optimization for clinical scenarios
- Modular, production-ready code structure

## Dataset

**Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (Cleveland database)

**Size:** 303 patients, 13 clinical features

**Features:**
- **age**: Age in years
- **sex**: Sex (1 = male, 0 = female)
- **cp**: Chest pain type (4 values)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (values 0,1,2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

**Target:** Binary classification (0 = no disease, 1 = disease present)

## Project Structure

```
cardiovascular-disease-prediction/
│
├── config.py                      # Configuration and hyperparameters
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── data/
│   ├── __init__.py
│   └── load_data.py              # Data loading and preprocessing
│
├── models/
│   ├── __init__.py
│   ├── logistic_model.py         # Logistic Regression
│   ├── random_forest_model.py    # Random Forest classifier
│   └── pca_model.py              # PCA + Logistic Regression
│
├── utils/
│   ├── __init__.py
│   ├── evaluation.py             # Metrics and evaluation functions
│   └── visualization.py          # Plotting and visualization
│
└── outputs/                       # Generated plots and results
    ├── model_comparison.png
    ├── metrics_comparison.png
    ├── roc_curves_comparison.png
    ├── confusion_matrix.png
    ├── confusion_matrix_optimized.png
    └── roc_curve_optimized.png
```

##  Installation & Usage

### Prerequisites
- Python 3.7+
- pip

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Tomisin777/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction
```

2. **Create a virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Model

**Run the complete pipeline:**
```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Train three models (Logistic Regression, PCA + Logistic, Random Forest)
3. Evaluate models 
4. Generate visualizations 
5. Perform threshold optimization
6. Save all results to the `outputs/` directory

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8852 | 0.8966 | 0.8387 | 0.8667 | 0.9468 |
| PCA + Logistic | 0.8361 | 0.8333 | 0.8065 | 0.8197 | 0.9121 |
| Random Forest | 0.8852 | 0.9310 | 0.8710 | 0.9000 | 0.9512 |

### Key Findings

- **Best Overall Model:** Random Forest achieves the highest ROC-AUC (0.9512) and F1-Score (0.9000)
- **Logistic Regression:** Strong baseline performance with good interpretability
- **PCA Analysis:** Performance drops with dimensionality reduction (13 → 5 components), suggesting all original features are valuable

### Threshold Optimization

Default classification threshold (0.5) prioritizes balanced accuracy. For clinical use cases where missing a disease case is more costly than a false alarm, we can adjust the threshold:

**Optimized Threshold (0.15):**
- **Sensitivity:** 100% (catches all disease cases)
- **Specificity:** ~50% (higher false positive rate)
- **Use case:** Screening tool where follow-up testing confirms diagnosis

## Visualizations

The project generates multiple visualizations:

1. **Model Comparison (Accuracy)** - Bar chart comparing model accuracies
2. **Comprehensive Metrics Comparison** - Multi-metric comparison across models
3. **ROC Curves** - Shows sensitivity vs specificity tradeoff for all models
4. **Confusion Matrices** - Visual breakdown of predictions vs actual values
5. **Optimized Model ROC** - ROC curve for threshold-optimized model

All plots are saved to the `outputs/` directory.

##  Configuration

Model hyperparameters and settings can be adjusted in `config.py`:

```python
RANDOM_STATE = 42              # For reproducibility
TEST_SIZE = 0.2                # Train/test split ratio
LOGISTIC_MAX_ITER = 1000       # Logistic regression iterations
PCA_COMPONENTS = 5             # Number of PCA components
OPTIMAL_THRESHOLD = 0.15       # Classification threshold
```

## Methodology

### Data Preprocessing
1. Handle missing values (marked as '?') in 'ca' and 'thal' columns
2. Convert data types to numeric
3. Impute missing values with column mean
4. Convert target to binary (0 = no disease, 1 = disease)
5. Split into 80% training, 20% testing

### Model Training
- **Logistic Regression:** Linear classifier, good baseline
- **Random Forest:** Ensemble method, captures non-linear patterns
- **PCA + Logistic:** Dimensionality reduction to test feature importance

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Precision:** % of positive predictions that are correct
- **Recall/Sensitivity:** % of actual positive cases detected
- **Specificity:** % of actual negative cases correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (0.5 = random, 1.0 = perfect)

## Limitations

1. **Small Dataset:** Only 303 samples - results may not generalize well
2. **Single Source:** Data from one location (Cleveland Clinic)
3. **Class Imbalance:** Not addressed in current implementation
4. **Missing Data:** Simple mean imputation may introduce bias
5. **Feature Engineering:** Limited exploration of derived features
6. **No External Validation:** Not tested on independent dataset

** IMPORTANT:** This is an **educational project** and should **NOT** be used for actual medical diagnosis or clinical decision-making.


### Technical Improvements for the future 
- [ ] Implement k-fold cross-validation for robust performance estimates
- [ ] Try advanced models 
- [ ] Explore feature engineering
- [ ] Test on external validation datasets

##  References

- UCI Machine Learning Repository: Heart Disease Dataset
- Scikit-learn Documentation


##  Author

**Tomi**
- GitHub: [@Tomisin777](https://github.com/Tomisin777)
- Field: Biomedical Engineering (Master's)
- Interests: Cardiovascular Health, Cardio-Oncology, AI in Healthcare

## What I Learned

This project was a way for me to apply what I’ve learned about machine learning in a biomedical context. Through it, I learned how to:

* Preprocess and clean real-world clinical data, handling missing and categorical values properly.
* Compare different supervised learning models — Logistic Regression, Random Forest, KNN, and SVM — to understand their strengths and trade-offs.
* Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
* Visualize results with ROC curves and confusion matrices to interpret model behavior more intuitively.
* Reflect on the importance of interpretability in healthcare ML — understanding *why* a prediction is made can be just as important as the prediction itself.



## Limitations

This project was created as a personal learning exercise and is **not intended for clinical use**. Some key limitations include:

* The dataset (UCI Cleveland Heart Disease) is small and may not generalize well to real-world populations.
* The models were trained and evaluated on a single dataset without external validation.
* Hyperparameter tuning and cross-validation were limited, so reported metrics should be viewed as exploratory.
* Feature importances and decision boundaries were interpreted in a simplified way for educational purposes.

Despite these limitations, the project gave me hands-on experience in building and evaluating ML models in the context of cardiovascular health — an area I’m passionate about exploring further.

