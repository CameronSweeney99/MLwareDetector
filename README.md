# MLwareDetector

MLwareDetector is a malware detection application developed in Python. It utilizes machine learning to analyze and predict whether a given executable file (.exe) is benign or malicious. The application features a user-friendly graphical interface and supports drag-and-drop functionality for easy file scanning.

## Features

- **Machine Learning Model**: Utilizes a RandomForestClassifier model trained on extensive malware and benign datasets.
- **Feature Extraction**: Implements feature extraction from PE (Portable Executable) files.
- **Graphical User Interface**: Built with Tkinter, providing a simple and intuitive interface.
- **Drag and Drop**: Users can drag files directly onto the application to initiate scanning.
- **File Upload**: Traditional file upload option is also available.
- **Data Exploration**: Includes a script for exploring the dataset, helping in understanding data distribution and feature importance.

## Installation

To run MLwareDetector, you need to have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

### Dependencies

Install the necessary Python packages using pip:

---bash
pip install -r requirements.txt

Running the Application
Navigate to the app directory and run the main script:

---bash
Copy code
cd app
python main.py
Usage
Upon launching MLwareDetector, you can either:

Drag and drop a .exe file onto the application window.
Click the 'Upload File' button and select a .exe file from your file system.
After file selection, the application will display the prediction result indicating whether the file is 'Malicious' or 'Benign'.

Feature Extraction
The extract_features.py script processes PE files to extract various features, including:

Binary properties like presence of resources, TLS, debug information.
Byte histograms and entropy of the file content.
Import characteristics, including specific DLLs and functions used.
Section details like size, entropy, and virtual size ratios.
Data Exploration
The data exploration script (data_exploration.py) is designed to analyze the datasets used for training the model. It provides insights into class distributions, missing values, and key feature statistics.

Development
This project includes separate modules for different functionalities:

extract_features.py: Handles the extraction of features from PE files.
load_detector.py: Manages loading the pre-trained machine learning model and making predictions.
main.py: The main application script with GUI components.

License
Apache-2.0

Contact
For any queries regarding MLwareDetector, please reach out via GitHub issues.