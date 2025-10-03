#Import libraries
import pandas as pd 
import numpy as np
#load dataset 
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None, names= ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

#cleaning 
# For ca column:
data['ca'] = data['ca'].replace('?', np.nan) #replace '?' with NaN
data['ca'] = pd.to_numeric(data['ca']) #change strings to numeric
data['ca'] = data['ca'].fillna(data['ca'].mean()) #fill NaNs with mean 
# For thal column 
data['thal'] = data['thal'].replace('?',np.nan)
data['thal'] = pd.to_numeric(data['thal'])
data['thal'] = data['thal'].fillna(data['thal'].mean())

#transform thal to binary 
data['target'] = (data['target'] > 0).astype(int)

#Split model 

from sklearn.model_selection import train_test_split
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(f"Training samples: {X_train.shape}")
#print(f"Testing samples: {X_test.shape}")

# Build model (logistic regression) 

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42, max_iter=1000 )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
#print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


#Using PCA 
from sklearn.decomposition import PCA 
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#train logstic regression on new pca data 
model_pca = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_pca, y_train)
y_pred_pca = model.predict(X_test_pca)
print(f"Original Model Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"PCA Model Accuracy: {accuracy_score(y_test, y_pred_pca)}")


# Trying Random Forest model 
from sklearn.ensemble import RandomForestClassifier

# Create and train
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")


#Creating Visuals 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix

#compare models
models = ['Logistic Regression', 'PCA + Logistic', 'Random Forest']
accuracies = [0.885, 0.836, 0.885]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim([0.7, 1.0])
#plt.savefig('model_comparison.png')
#plt.show()

#Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize =(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('confusion_matrix.png')
plt.show()