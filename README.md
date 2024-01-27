# MLwareDetector

MLwareDetector is a malware detection application developed in Python. It utilizes machine learning to analyze and predict whether a given executable file (.exe) is benign or malicious. The application features a user-friendly graphical interface and supports drag-and-drop functionality for easy file scanning.

## Features

- **Intuitive GUI**: The application includes a user-friendly graphical interface that supports both traditional file upload and drag-and-drop functionalities for scanning files.
- **Robust ML Model**: Employs a RandomForestClassifier model, providing high accuracy in distinguishing between malware and benign files.
- **Feature Extraction**: Utilizes the `feature_extraction.py` script to extract meaningful attributes from PE files which are critical for machine learning model training.
- **Data Exploration**: Comes with a `data_exploration.py` script that offers insights into the dataset through visualizations of feature distributions and class imbalances.
- **Versatile Model Training**: Incorporates different training scripts (`train_decision_tree.py`, `train_neural_network.py`, etc.) for experimenting with various machine learning algorithms.
- **Comprehensive Evaluation**: Uses `test_random_forest.py` and other test scripts to assess the performance of models using metrics like accuracy, ROC curve, and confusion matrix.

## Installation

To run MLwareDetector, you need to have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

### Dependencies

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```
### Running the Application
Navigate to the app directory and run the main script:

```bash
cd app
python main.py
```
### Usage
Upon launching MLwareDetector, you can either:

Drag and drop a .exe file onto the application window.
Click the 'Upload File' button and select a .exe file from your file system.
After file selection, the application will display the prediction result indicating whether the file is 'Malicious' or 'Benign'.

## Feature Extraction
The extract_features.py script processes PE files to extract various features, including:

Binary properties like presence of resources, TLS, debug information.
Byte histograms and entropy of the file content.
Import characteristics, including specific DLLs and functions used.
Section details like size, entropy, and virtual size ratios.
Data Exploration
The data exploration script (`data_exploration.py`) is designed to analyze the datasets used for training the model. It provides insights into class distributions, missing values, and key feature statistics.

## Development

This section details the various components and their functionalities within the MLwareDetector project.

### `explore_data.py`
This script performs exploratory data analysis on the datasets. It visualizes missing values, class distributions, and feature distributions to provide insights into the data used for training the models.

### `feature_extraction.py`
Processes PE files to extract several features, including binary properties, byte histograms, import characteristics, section details, and more. These features are pivotal for the machine learning model to make accurate predictions.

### `model_selection.py`
Compares different machine learning models using cross-validation. It helps in selecting the best model based on performance metrics like ROC-AUC scores.

### `split_data.py`
Splits the dataset into training, validation, and test sets. It ensures that the models are evaluated on an independent test set that was not seen during the training phase.

### `test_random_forest.py`
Evaluates the RandomForestClassifier model on the test dataset. It outputs the accuracy, classification report, and plots the confusion matrix.

### `train_decision_tree.py`, `train_neural_network.py`, `train_random_forest.py`
These scripts are responsible for training different types of models. They handle the entire training process, including data loading, model fitting, and saving the trained model.

## License
Apache-2.0

## Contact
For any queries regarding MLwareDetector, please reach out via GitHub issues.