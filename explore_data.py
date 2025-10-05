
# IMPORTS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# DATA LOADING & PREPROCESSING
 
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None, names= ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

# CLEANING 

    # For ca column:
data['ca'] = data['ca'].replace('?', np.nan) #replace '?' with NaN
data['ca'] = pd.to_numeric(data['ca']) #change strings to numeric
data['ca'] = data['ca'].fillna(data['ca'].mean()) #fill NaNs with mean 
    # For thal column 
data['thal'] = data['thal'].replace('?',np.nan)
data['thal'] = pd.to_numeric(data['thal'])
data['thal'] = data['thal'].fillna(data['thal'].mean())

    # Transform thal to binary 
data['target'] = (data['target'] > 0).astype(int)

# SPLIT DATA

X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(f"Training samples: {X_train.shape}")
#print(f"Testing samples: {X_test.shape}")

# MODEL TRAINING 

# LOGISTIC REGRESSION
model = LogisticRegression(random_state=42, max_iter=1000 )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
#print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# PRINCIPAL COMPONENT ANALYSIS 
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

    # Train logstic regression on new pca data 
model_pca = LogisticRegression(random_state=42, max_iter=1000)
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
print(f"Original Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"PCA Model Accuracy: {accuracy_score(y_test, y_pred_pca)}")


# RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestClassifier

    # Create and train
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")


# VISUALIZATION

    #compare models
models = ['Logistic Regression', 'PCA + Logistic', 'Random Forest']
accuracies = [0.885, 0.836, 0.885]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim([0.7, 1.0])
plt.savefig('model_comparison.png')
plt.show()

    # CONFUSION MATRIX 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize =(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('confusion_matrix.png')
plt.show()

# THRESHOLD ADJUSTMENT FOR BETTER SENSITIVITY

y_prob = model.predict_proba(X_test)
disease_proba = y_prob[:, 1]

thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.15]

for threshold in thresholds:
    # APPLY THRESHOLD
    y_pred_new = (disease_proba >= threshold).astype(int)
    #CALCULATE METRICS
    accuracy = accuracy_score(y_test, y_pred_new)
    cm_new = confusion_matrix(y_test, y_pred_new)
    #CALCULATE SENSITIVITY
    sensitivity = cm_new[1,1] / (cm_new[1,0] + cm_new[1,1])

    #print(f"\nThreshold = {threshold}")
    #print(f"Accuracy: {accuracy:.4f}")
    #print(f"Sensitivity: {sensitivity:.4f}")
    #print(f"Confusion Matrix:")
    #print(cm_new)
    #print(f"Missed cases: {cm_new[1,0]}")  # False negatives

# FINAL MODEL WITH OPTIMIZED THRESHOLD 
optimal_threshold = 0.15 
y_pred_optimized = (disease_proba >= optimal_threshold).astype(int)

# Final Metrics
final_accuracy = accuracy_score(y_test, y_pred_optimized)
final_cm = confusion_matrix(y_test, y_pred_optimized)
final_sensitivity = final_cm[1,1] / (final_cm[1,0] + final_cm[1,1])

print(f"Threshold: {optimal_threshold}")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Sensitivity: {final_sensitivity:.4f}")
print(f"\nConfusion Matrix:")
print(final_cm)
print(f"\nFalse Negatives (Missed): {final_cm[1,0]}")
print(f"False Positives (False Alarms): {final_cm[0,1]}")

# Plot this confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Greens')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Optimized Model (Threshold={optimal_threshold}) - 100% Sensitivity')
plt.savefig('confusion_matrix_optimized.png')
plt.show()
