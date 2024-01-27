import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import joblib
import extract_features
import numpy as np

class MLwareDetectorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title('MLwareDetector')
        self.geometry('500x300') 
        self.configure(bg='#2a2a2e')

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = tk.Frame(self, bg='#2a2a2e')
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        self.label = tk.Label(self.main_frame, text="MLwareDetector", font=("Arial", 20, "bold"), bg='#2a2a2e', fg='white')
        self.label.pack(pady=20)

        self.upload_button = tk.Button(self.main_frame, text='Upload File', command=self.select_file, bg='#4a7abc', fg='white', font=("Arial", 12))
        self.upload_button.pack(pady=10)

        self.status_label = tk.Label(self.main_frame, text="", font=("Arial", 12), bg='#2a2a2e', fg='white')
        self.status_label.pack(pady=20)

        # Enable drag and drop
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
            self.status_label.config(text=f'File selected: {filename}')
            try:
                model = joblib.load('malware_detector.joblib')
                features = extract_features.encode_pe_file(filename)
                if features is not None:
                    features = np.array([features])
                    result = model.predict(features)[0]
                    prediction_text = "Malware" if result == 1 else "Benign"
                    self.status_label.config(text=f'File: {filename}\nPrediction: {prediction_text}')
                else:
                    self.status_label.config(text="Error in feature extraction")
            except Exception as e:
                self.status_label.config(text=f"An error occurred: {e}")

if __name__ == "__main__":
    app = MLwareDetectorApp()
    app.mainloop()
