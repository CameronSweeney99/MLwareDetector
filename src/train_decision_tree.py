#train_decision_tree.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import tensorflow as tf

# Check if GPU is available
print("Is GPU available:", tf.test.is_gpu_available())

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def load_data(filename):
    # Load the data from the file and separate into features and target label
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1]  # all columns except the last one
    y = data.iloc[:, -1]  # the last column
    return X, y

def main():
    # Load datasets
    X_train, y_train = load_data('../data/training_extracted_features.csv')
    X_val, y_val = load_data('../data/validation_extracted_features.csv')
    X_test, y_test = load_data('../data/test_extracted_features.csv')

    # Initialize the model
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_val, y_val, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", cv_scores.mean())

    # Evaluate the model on the test data
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("\nTest Accuracy:", test_accuracy)
    print("\nClassification Report:\n", classification_report(y_test, test_predictions))

    # Save the model
    dump(model, '../models/decision_tree.joblib')
    print("Model saved in 'model_folder/my_model.joblib'")

if __name__ == "__main__":
    main()
