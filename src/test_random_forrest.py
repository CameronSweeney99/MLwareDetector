#split_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def load_data(filename):
    print(f"Loading test data from {filename}...")
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1] 
    
    # Convert labels to numeric
    label_mapping = {'benign': 0, 'malware': 1}
    y = data.iloc[:, -1].map(label_mapping)
    
    print("Test data loaded successfully.")
    return X, y

def plot_confusion_matrix(conf_mat, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    # Load the pre-trained model
    print("Loading the pre-trained Random Forest model...")
    model = load('../models/random_forest.joblib')
    print("Model loaded successfully.")

    # Load test data
    X_test, y_test = load_data('../data/test_extracted_features.csv')

    # Predict using the model
    print("Predicting labels on the test data...")
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("\nTest Accuracy:", test_accuracy)
    print("\nClassification Report:\n", classification_report(y_test, test_predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)
    print("\nConfusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=['Benign', 'Malware'])
    plt.savefig('C:/Users/cammy/Documents/GitHub/MLwareDetector/plots/random_forrest/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()
