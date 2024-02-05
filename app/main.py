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

    def select_file(self):
        filename = filedialog.askopenfilename()
        self.process_file(filename)

    def drop(self, event):
        if event.data:
            files = self.main_frame.tk.splitlist(event.data)
            for f in files:
                self.process_file(f)

    def process_file(self, filename):
        if filename:
            self.status_label.config(text=f'Loading file: {filename}')
            self.update_idletasks()

            threading.Thread(target=self.scan_file, args=(filename,), daemon=True).start()

    def compute_and_plot_shap(self, model, features, filename):
        # Assuming features is a numpy array; convert it to a DataFrame with column names
        if isinstance(features, np.ndarray):
            features_df = pd.DataFrame(features.reshape(1, -1), columns=get_feature_names())
        else:
            features_df = features

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)

        # Plotting
        plt.figure()
        shap.summary_plot(shap_values, features_df, show=False)
        plot_path = "shap_summary.png"
        plt.savefig(plot_path)
        plt.close()

        # Display the plot
        self.show_shap_plot(plot_path)

    def show_shap_plot(self, plot_path):
        # Create a new top-level window
        plot_window = tk.Toplevel(self)
        plot_window.title("SHAP Summary Plot")

        # Load and display the SHAP plot image in the new window
        shap_image = Image.open(plot_path)
        shap_photo = ImageTk.PhotoImage(shap_image)

        # Use a label to display the image
        plot_label = tk.Label(plot_window, image=shap_photo)
        plot_label.pack()

        # Keep a reference to the photo object to prevent it from being garbage collected
        plot_label.image = shap_photo

    def finalize_prediction(self, filename, model):
        features = extract_features.encode_pe_file(filename)
        if features is not None:
            features = np.array([features])
            result = model.predict(features)[0]
            prediction_text = "Malware Detected" if result == 1 else "File is Benign"
            self.status_label.config(text=f'Result: {prediction_text}')
            # Call SHAP computation and plotting method
            self.compute_and_plot_shap(model, features, filename)
        else:
            self.status_label.config(text="Error in feature extraction")
       
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
