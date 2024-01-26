#explore_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to load data
def load_data(file_paths):
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Function to perform basic exploratory data analysis
def explore_data(data, label_column_name, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Print basic info
    print("Basic Information:")
    print(data.info())

    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)

    # Only plot missing values if there are any
    if missing_values.sum() > 0:
        plt.figure(figsize=(10, 6))
        missing_values[missing_values > 0].plot(kind='bar')
        plt.title('Missing Values per Feature')
        plt.ylabel('Number of Missing Values')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'missing_values.png'))
        plt.close()
    else:
        print("No missing values to plot.")

    # Class distribution
    class_distribution = data[label_column_name].value_counts()
    print("\nClass Distribution:")
    print(class_distribution)

    # Save class distribution plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x=label_column_name, data=data)
    plt.title('Class Distribution')
    plt.savefig(os.path.join(output_folder, 'class_distribution.png'))
    plt.close()

    # Feature distribution plots for a selected number of features
    top_n_features = 10
    feature_vars = data.var().sort_values(ascending=False)
    top_features = feature_vars.head(top_n_features).index.tolist()

    # Plot histograms for top features
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Distribution of Top Feature: {feature}')
        plt.savefig(os.path.join(output_folder, f'distribution_{feature}.png'))
        plt.close()

# Paths to the datasets
training_path = '../data/training_extracted_features.csv'
validation_path = '../data/validation_extracted_features.csv'
test_path = '../data/test_extracted_features.csv'

# Load and combine datasets
all_data = load_data([training_path, validation_path, test_path])

# Convert labels to numeric if they're not already
label_column_name = 'label'  # change to your actual label column name if different
all_data[label_column_name] = all_data[label_column_name].map({'benign': 0, 'malware': 1})

# Perform EDA and save plots
explore_data(all_data, label_column_name, '../plots/data_exploration/')
