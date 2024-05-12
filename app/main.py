#main.py
import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from extract_features import get_feature_names
import joblib
import extract_features
import numpy as np
import pefile
import threading
import os
import matplotlib.pyplot as plt
import shap
import pandas as pd

class MLwareDetectorApp(TkinterDnD.Tk):
    # Initialization of the main application window
    def __init__(self):
        super().__init__()
        self.title('MLwareDetector')
        self.geometry('800x500')
        self.configure(bg='#f0f0f0')

        self.iconbitmap('V1_Symbol.ico')
        pil_image = Image.open('V1_Colour.png')
        pil_image = pil_image.resize((200, 200), Image.Resampling.LANCZOS)  
        self.logo_image = ImageTk.PhotoImage(pil_image)

        self.create_widgets()

    # Creates the GUI components
    def create_widgets(self):
        self.main_frame = tk.Frame(self, bg='#f0f0f0')  
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        self.logo_label = tk.Label(self.main_frame, image=self.logo_image, bg='#f0f0f0')  
        self.logo_label.pack(pady=20)

        self.upload_button = tk.Button(self.main_frame, text='Upload File', command=self.select_file, bg='#3babc9', fg='white', font=("Arial", 12))
        self.upload_button.pack(pady=10)

        self.status_label = tk.Label(self.main_frame, text="", font=("Arial", 14), bg='#f0f0f0', fg='black')  
        self.status_label.pack(pady=20)

        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop)

    # Handles file selection via file dialog
    def select_file(self):
        filename = filedialog.askopenfilename()
        self.process_file(filename)

    # Handles file drop operation
    def drop(self, event):
        if event.data:
            files = self.main_frame.tk.splitlist(event.data)
            for f in files:
                self.process_file(f)

    # Starts processing the selected file
    def process_file(self, filename):
        if filename:
            self.status_label.config(text=f'Loading file: {filename}')
            self.update_idletasks()

            threading.Thread(target=self.scan_file, args=(filename,), daemon=True).start()

    # Computes SHAP values and generates a plot
    def compute_and_plot_shap(self, model, features, filename):
        if isinstance(features, np.ndarray):
            features_df = pd.DataFrame(features.reshape(1, -1), columns=get_feature_names())
        else:
            features_df = features

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)

        plt.figure()
        shap.summary_plot(shap_values, features_df, show=False)
        plot_path = "shap_summary.png"
        plt.savefig(plot_path)
        plt.close()

        self.show_shap_plot(plot_path)

    # Displays the generated SHAP plot
    def show_shap_plot(self, plot_path):
        plot_window = tk.Toplevel(self)
        plot_window.title("SHAP Summary Plot")

        shap_image = Image.open(plot_path)
        shap_photo = ImageTk.PhotoImage(shap_image)

        plot_label = tk.Label(plot_window, image=shap_photo)
        plot_label.pack()

        plot_label.image = shap_photo

    # Finalizes the prediction process
    def finalize_prediction(self, filename, model):
        features = extract_features.encode_pe_file(filename)
        if features is not None:
            features = np.array([features])
            result = model.predict(features)[0]
            prediction_text = "Malware Detected" if result == 1 else "File is Benign"
            self.status_label.config(text=f'Result: {prediction_text}')
            self.compute_and_plot_shap(model, features, filename)
        else:
            self.status_label.config(text="Error in feature extraction")
       
    # Scans the file using the provided model
    def scan_file(self, filename):
        if not os.path.isfile(filename):
            self.status_label.config(text="Error: File does not exist.")
            return

        try:
            pe = pefile.PE(filename)
        except pefile.PEFormatError:
            self.status_label.config(text="Error: Not a valid PE file.")
            return
        except Exception as e:
            self.status_label.config(text=f"An error occurred: {e}")
            return

        self.status_label.config(text='Processing file, please wait...')
        try:
            model = joblib.load('malware_detector.joblib')
            self.status_label.config(text='Dataset loaded. Scanning now...')
            self.after(0, self.finalize_prediction, filename, model)
        except Exception as e:
            self.status_label.config(text=f"An error occurred: {e}")

if __name__ == "__main__":
    app = MLwareDetectorApp()
    app.mainloop()
