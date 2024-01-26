#model_selection.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Constants
DATA_PATH = '../data/test_extracted_features.csv'  

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Prepare the data
X = data.drop(['label'], axis=1).select_dtypes(include=[np.number])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addressing class imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Initialize Dummy Classifier
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)

# Train the Dummy Classifier
dummy_clf.fit(X_train_balanced, y_train_balanced)

# Evaluate the Dummy Classifier
dummy_predictions = dummy_clf.predict(X_test_scaled)
print("Dummy Classifier Report")
print(classification_report(y_test, dummy_predictions, zero_division=0))  # Modified line
print("ROC-AUC:", roc_auc_score(y_test, dummy_clf.predict_proba(X_test_scaled)[:, 1]))

# Initialize models
models = {
    'LogisticRegression': LogisticRegression(max_iter=10000),
    'DecisionTree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}

# Stratified K-Fold for better cross-validation
cv = StratifiedKFold(n_splits=5)

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='roc_auc')
    print(f"Cross-validated ROC-AUC scores: {cv_scores}")
    print(f"Mean ROC-AUC score: {np.mean(cv_scores)}")
    
    # Training and testing on the split
    model.fit(X_train_balanced, y_train_balanced)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities)}\n")
