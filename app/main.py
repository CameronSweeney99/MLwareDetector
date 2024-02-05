import tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import joblib
import extract_features
import numpy as np

class MLwareDetectorApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title('MLwareDetector')
        self.geometry('800x500')
        self.configure(bg='#f0f0f0')

        self.iconbitmap('V1_Symbol.ico')
        # Load the logo image using PIL
        pil_image = Image.open('V1_Colour.png')
        # Resize the image using the PIL.Image.Resampling.LANCZOS filter for better quality
        pil_image = pil_image.resize((200, 200), Image.Resampling.LANCZOS)  # Adjust the size (100, 100) as needed
        self.logo_image = ImageTk.PhotoImage(pil_image)

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = tk.Frame(self, bg='#f0f0f0')  # Changed to very light grey
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Create a label for the logo image
        self.logo_label = tk.Label(self.main_frame, image=self.logo_image, bg='#f0f0f0')  # Changed to very light grey
        self.logo_label.pack(pady=20)

        self.upload_button = tk.Button(self.main_frame, text='Upload File', command=self.select_file, bg='#4a7abc', fg='white', font=("Arial", 12))
        self.upload_button.pack(pady=10)

        self.status_label = tk.Label(self.main_frame, text="", font=("Arial", 12), bg='#f0f0f0', fg='black')  # Changed to very light grey and text to black for contrast
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
