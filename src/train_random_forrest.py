#train_random_forrest.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def load_data(filename):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1].map({'benign': 0, 'malware': 1}) 
    print("Data loaded successfully.")
    return X, y

def plot_confusion_matrix(cm, classes, plot_directory, title, dataset_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{title} - {dataset_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{plot_directory}/confusion_matrix_{dataset_name}.png')
    plt.show()

def plot_top_feature_importances(model, columns, top_n=20, plot_directory='../plots/random_forrest'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 10))
    plt.title('Top Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f'{plot_directory}/top_features.png')
    plt.show()

def main():
    start_time = time.time()
    X_train, y_train = load_data('../data/training_extracted_features.csv')
    X_val, y_val = load_data('../data/validation_extracted_features.csv')
    X_test, y_test = load_data('../data/test_extracted_features.csv')

    model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_val, y_val, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average cross-validation score: {cv_scores.mean()}")

    # OOB Score
    print("OOB Score:", model.oob_score_)

    # Confusion Matrices for each dataset
    cm_train = confusion_matrix(y_train, model.predict(X_train))
    plot_confusion_matrix(cm_train, ['Benign', 'Malware'], '../plots/random_forrest', 'Confusion Matrix', 'Training')

    cm_val = confusion_matrix(y_val, model.predict(X_val))
    plot_confusion_matrix(cm_val, ['Benign', 'Malware'], '../plots/random_forrest', 'Confusion Matrix', 'Validation')

    cm_test = confusion_matrix(y_test, model.predict(X_test))
    plot_confusion_matrix(cm_test, ['Benign', 'Malware'], '../plots/random_forrest', 'Confusion Matrix', 'Test')

    # Feature importances
    plot_top_feature_importances(model, X_train.columns, top_n=20)

    # Permutation Feature Importance
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(12, 8))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_val.columns[sorted_idx])
    plt.title("Permutation Importances (validation set)")
    plt.tight_layout()
    plt.savefig('../plots/random_forrest/permutation_importances.png')
    plt.show()

    dump(model, '../models/random_forest.joblib')
    print("Model saved in '../models/random_forest.joblib'")

    elapsed_time = time.time() - start_time
    print(f"Total process completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
