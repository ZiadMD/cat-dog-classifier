import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import threading

class DLModelHandler:
    def __init__(self):
        self.model = None
        self.input_size = (256, 256)  # Default for common models
        self.loaded_model_path = None

    def load_model(self, model_path):
        """Load Keras model from .h5 file with validation"""
        try:
            self.model = load_model(model_path)
            self.loaded_model_path = model_path
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Model loading failed: {str(e)}"

    def preprocess_image(self, image_path):
        """Process image for model input with error handling"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.input_size)
            arr = np.array(img).astype('float32') / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            return True, np.expand_dims(arr, axis=0)
        except Exception as e:
            return False, f"Image processing error: {str(e)}"

    def predict(self, processed_img):
        """Make prediction with confidence score"""
        try:
            predictions = self.model.predict(processed_img)
            pred_idx = np.argmax(predictions)
            confidence = predictions[0][pred_idx]
            return True, (pred_idx, float(confidence))
        except Exception as e:
            return False, f"Prediction failed: {str(e)}"

class ImageViewer(tk.Frame):
    def __init__(self, parent, max_size=(400, 400)):
        super().__init__(parent)
        self.max_size = max_size
        self.image_label = tk.Label(self)
        self.image_label.pack()

    def display_image(self, image_path):
        """Display image with aspect ratio preservation"""
        try:
            img = Image.open(image_path)
            img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference
            return True
        except Exception as e:
            return False

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DL Model Tester")
        self.geometry("800x600")
        self.model_handler = DLModelHandler()
        self.create_widgets()
        self.setup_threading()

    def create_widgets(self):
        """Create and arrange UI components"""
        # File Selection Panel
        self.file_frame = ttk.LabelFrame(self, text="File Selection")
        self.file_frame.pack(pady=10, fill='x')

        self.model_btn = ttk.Button(self.file_frame, text="Load Model", command=self.load_model)
        self.model_btn.pack(side='left', padx=5)
        
        self.image_btn = ttk.Button(self.file_frame, text="Load Image", command=self.load_image)
        self.image_btn.pack(side='left', padx=5)

        # Image Preview
        self.viewer = ImageViewer(self)
        self.viewer.pack(pady=10, expand=True, fill='both')

        # Prediction Controls
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(pady=10)
        
        self.predict_btn = ttk.Button(
            self.control_frame, 
            text="Run Prediction", 
            command=self.start_prediction_thread
        )
        self.predict_btn.pack(side='left', padx=5)

        # Results Display
        self.results = tk.Text(self, height=4, width=50)
        self.results.pack(pady=10)
        self.results.insert('end', "Predictions will appear here...")

        # Status Bar
        self.status = ttk.Label(self, text="Ready", relief='sunken')
        self.status.pack(side='bottom', fill='x')

    def setup_threading(self):
        self.prediction_thread = None
        self.prediction_running = False

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Keras Models", "*.h5")])
        if path:
            success, message = self.model_handler.load_model(path)
            self.status.config(text=message)
            self.results.delete(1.0, 'end')
            self.results.insert('end', f"Model: {path.split('/')[-1]} loaded")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Images", "*.jpg *.jpeg *.png"), ("All Files", "*.*")
        ])
        if path and self.viewer.display_image(path):
            self.current_image = path
            self.status.config(text=f"Image loaded: {path.split('/')[-1]}")
            self.results.delete(1.0, 'end')

    def start_prediction_thread(self):
        """Handle prediction in separate thread to prevent UI freeze"""
        if not self.prediction_running:
            self.prediction_running = True
            self.predict_btn.config(state='disabled')
            self.status.config(text="Processing...")
            self.prediction_thread = threading.Thread(target=self.run_prediction)
            self.prediction_thread.start()
            self.check_thread_status()

    def check_thread_status(self):
        """Monitor prediction thread completion"""
        if self.prediction_thread.is_alive():
            self.after(100, self.check_thread_status)
        else:
            self.prediction_running = False
            self.predict_btn.config(state='normal')

    def run_prediction(self):
        """Main prediction logic"""
        if not hasattr(self, 'current_image'):
            self.status.config(text="No image selected!")
            return

        success, processed_img = self.model_handler.preprocess_image(self.current_image)
        if not success:
            self.show_error(processed_img)
            return

        success, prediction = self.model_handler.predict(processed_img)
        if success:
            self.show_prediction(*prediction)
        else:
            self.show_error(prediction)

    def show_prediction(self, class_idx, confidence):
        """Update UI with prediction results"""
        self.results.delete(1.0, 'end')
        self.results.insert('end', 
            f"Predicted class: {class_idx}\nConfidence: {confidence:.2%}")
        self.status.config(text="Prediction complete")

    def show_error(self, message):
        """Display error messages consistently"""
        self.status.config(text="Error occurred!")
        self.results.delete(1.0, 'end')
        self.results.insert('end', f"Error: {message}")

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()