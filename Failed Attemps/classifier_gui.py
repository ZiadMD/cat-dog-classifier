#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import json

import numpy as np
from PIL import Image
import tensorflow as tf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QSpinBox, QDoubleSpinBox, QMessageBox)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage


class ImageHandler:
    """Handles image loading, preprocessing and preview generation."""
    
    def __init__(self):
        self.image: Optional[Image.Image] = None
        self.image_path: Optional[str] = None
        
    def load_image(self, image_path: str) -> bool:
        """Load an image from path and return success status."""
        try:
            self.image = Image.open(image_path)
            self.image_path = image_path
            return True
        except Exception as e:
            self.image = None
            self.image_path = None
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def preprocess_image(self, target_size: Tuple[int, int],
                        mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """Preprocess the loaded image for model input."""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Resize and convert to RGB
        img = self.image.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    
    def get_preview(self, max_size: Tuple[int, int] = (300, 300)) -> QPixmap:
        """Generate a preview QPixmap of the loaded image."""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Create a copy for preview
        preview = self.image.copy()
        preview.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert PIL image to QPixmap
        img_array = np.array(preview)
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_array.data, width, height, bytes_per_line,
                      QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img)


class ModelHandler:
    """Handles TensorFlow model loading and inference."""
    
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.model_path: Optional[str] = None
        
    def load_model(self, model_path: str) -> bool:
        """Load a TensorFlow model from path and return success status."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            return True
        except Exception as e:
            self.model = None
            self.model_path = None
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def get_input_shape(self) -> Optional[Tuple[int, int]]:
        """Return the expected input shape (height, width) for the model."""
        if self.model is None:
            return None
        
        input_shape = self.model.input_shape
        if input_shape is None or len(input_shape) != 4:
            return None
            
        return (input_shape[1], input_shape[2])
    
    def predict(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed image data."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        return self.model.predict(preprocessed_image)


class PredictionEngine:
    """Manages label mapping and prediction formatting."""
    
    def __init__(self):
        self.labels = ["Dog", "Cat"]  # Default labels
        
    def load_labels(self, labels_path: str) -> bool:
        """Load custom labels from a file."""
        try:
            ext = Path(labels_path).suffix.lower()
            if ext == '.txt':
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
            elif ext == '.json':
                with open(labels_path, 'r') as f:
                    self.labels = json.load(f)
            else:
                raise ValueError(f"Unsupported label file format: {ext}")
            
            if len(self.labels) != 2:
                raise ValueError("Expected exactly 2 labels")
                
            return True
        except Exception as e:
            self.labels = ["Cat", "Dog"]  # Reset to defaults
            raise ValueError(f"Failed to load labels: {str(e)}")
    
    def format_prediction(self, prediction: np.ndarray) -> str:
        """Format model output into human-readable prediction."""
        # if prediction.shape[-1] != 2:
        #     raise ValueError("Expected 2-class prediction")
            
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx] * 100
        
        return f"Prediction: {self.labels[class_idx]} ({confidence:.1f}%)"


class MainWindow(QMainWindow):
    """Main application window handling UI and logic coordination."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize handlers
        self.image_handler = ImageHandler()
        self.model_handler = ModelHandler()
        self.prediction_engine = PredictionEngine()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Cat vs Dog Classifier')
        self.setMinimumSize(600, 400)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create UI elements
        self.create_file_controls(layout)
        self.create_preview_area(layout)
        self.create_parameter_controls(layout)
        self.create_prediction_area(layout)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
    def create_file_controls(self, parent_layout: QVBoxLayout):
        """Create file selection controls."""
        file_layout = QHBoxLayout()
        
        # Image selection
        self.image_btn = QPushButton('Browse Image...')
        self.image_btn.clicked.connect(self.browse_image)
        file_layout.addWidget(self.image_btn)
        
        # Model selection
        self.model_btn = QPushButton('Browse Model...')
        self.model_btn.clicked.connect(self.browse_model)
        file_layout.addWidget(self.model_btn)
        
        parent_layout.addLayout(file_layout)
        
    def create_preview_area(self, parent_layout: QVBoxLayout):
        """Create image preview area."""
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }")
        parent_layout.addWidget(self.preview_label)
        
    def create_parameter_controls(self, parent_layout: QVBoxLayout):
        """Create preprocessing parameter controls."""
        param_layout = QHBoxLayout()
        
        # Image dimensions
        dim_layout = QVBoxLayout()
        dim_layout.addWidget(QLabel('Image Dimensions:'))
        
        dim_controls = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 1024)
        self.width_spin.setValue(224)
        dim_controls.addWidget(QLabel('Width:'))
        dim_controls.addWidget(self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 1024)
        self.height_spin.setValue(224)
        dim_controls.addWidget(QLabel('Height:'))
        dim_controls.addWidget(self.height_spin)
        
        dim_layout.addLayout(dim_controls)
        param_layout.addLayout(dim_layout)
        
        # Normalization parameters
        norm_layout = QVBoxLayout()
        norm_layout.addWidget(QLabel('Normalization:'))
        
        norm_controls = QHBoxLayout()
        self.mean_spin = QDoubleSpinBox()
        self.mean_spin.setRange(-255, 255)
        self.mean_spin.setValue(127.5)
        norm_controls.addWidget(QLabel('Mean:'))
        norm_controls.addWidget(self.mean_spin)
        
        self.std_spin = QDoubleSpinBox()
        self.std_spin.setRange(0.1, 255)
        self.std_spin.setValue(127.5)
        norm_controls.addWidget(QLabel('Std:'))
        norm_controls.addWidget(self.std_spin)
        
        norm_layout.addLayout(norm_controls)
        param_layout.addLayout(norm_layout)
        
        parent_layout.addLayout(param_layout)
        
    def create_prediction_area(self, parent_layout: QVBoxLayout):
        """Create prediction display area."""
        self.prediction_label = QLabel('No prediction yet')
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_label.setStyleSheet(
            "QLabel { font-size: 14pt; margin: 10px; }")
        parent_layout.addWidget(self.prediction_label)
        
    def browse_image(self):
        """Handle image file selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg);;All Files (*.*)")
        
        if file_path:
            self.load_image(file_path)
            
    def browse_model(self):
        """Handle model file/directory selection."""
        # Create filter for file dialog
        file_filter = "Model Files (*.h5 *.keras);;SavedModel Directory ();;All Files (*.*)"
        
        # First try to get a file (.h5 or .keras)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", file_filter)
            
        # If no file selected, try to get a directory (for SavedModel)
        if not file_path:
            file_path = QFileDialog.getExistingDirectory(
                self, "Select SavedModel Directory")
        
        if file_path:
            self.load_model(file_path)
            
    def load_image(self, image_path: str):
        """Load and display an image."""
        try:
            self.image_handler.load_image(image_path)
            self.update_preview()
            self.run_prediction()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def load_model(self, model_path: str):
        """Load a model and update UI accordingly."""
        try:
            # Check if it's a valid model path
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")
                
            self.model_handler.load_model(model_path)
            
            # Update input dimensions if available
            if (input_shape := self.model_handler.get_input_shape()):
                self.height_spin.setValue(input_shape[0])
                self.width_spin.setValue(input_shape[1])
                
            self.run_prediction()
            
            # Show success message
            QMessageBox.information(self, "Success", f"Model loaded successfully from:\n{model_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def update_preview(self):
        """Update the image preview."""
        try:
            pixmap = self.image_handler.get_preview()
            self.preview_label.setPixmap(pixmap)
        except Exception as e:
            self.preview_label.clear()
            QMessageBox.warning(self, "Warning", str(e))
            
    def run_prediction(self):
        """Run model prediction on current image."""
        try:
            if (self.image_handler.image is None or
                self.model_handler.model is None):
                return
                
            # Preprocess image
            preprocessed = self.image_handler.preprocess_image(
                (self.width_spin.value(), self.height_spin.value()),
                self.mean_spin.value(), self.std_spin.value())
            
            # Run prediction
            prediction = self.model_handler.predict(preprocessed)
            
            # Display result
            result_text = self.prediction_engine.format_prediction(prediction)
            self.prediction_label.setText(result_text)
            
        except Exception as e:
            self.prediction_label.setText("Prediction failed")
            QMessageBox.warning(self, "Prediction Error", str(e))
            
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for drag-and-drop."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        """Handle drop events for drag-and-drop."""
        urls = event.mimeData().urls()
        if not urls:
            return
            
        file_path = urls[0].toLocalFile()
        if not file_path:
            return
            
        # Check if it's an image file
        ext = Path(file_path).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            self.load_image(file_path)
        elif ext in ['.h5', '.keras'] or not ext:  # No ext could be SavedModel dir
            self.load_model(file_path)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main() 