import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QSpinBox,
                              QSlider, QProgressBar, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import qdarkstyle
import tensorflow as tf
import numpy as np
from keras_self_attention import SeqWeightedAttention as OriginalSeqWeightedAttention

class SeqWeightedAttention(OriginalSeqWeightedAttention):
    def __init__(self, use_bias=None, return_attention=None, **kwargs):
        super().__init__(use_bias=use_bias, return_attention=return_attention, **kwargs)

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        self.setMinimumSize(800, 600)
        
        # Initialize instance variables
        self.model = None
        self.current_image = None
        self.image_path = None
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for image
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #2a2a2a; border: 1px solid #3a3a3a; }")
        left_layout.addWidget(self.image_label)
        
        # Load image button
        load_image_btn = QPushButton("Load Image")
        load_image_btn.clicked.connect(self.load_image)
        left_layout.addWidget(load_image_btn)
        
        main_layout.addWidget(left_panel)
        
        # Create right panel for controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Model loading section
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        right_layout.addWidget(load_model_btn)
        
        # Input size controls
        size_layout = QHBoxLayout()
        
        width_label = QLabel("Width:")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 1024)
        self.width_spin.setValue(224)
        
        height_label = QLabel("Height:")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 1024)
        self.height_spin.setValue(224)
        
        size_layout.addWidget(width_label)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(height_label)
        size_layout.addWidget(self.height_spin)
        right_layout.addLayout(size_layout)
        
        # Threshold control
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_value_label = QLabel("50%")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value_label.setText(f"{v}%"))
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        right_layout.addLayout(threshold_layout)
        
        # Predict button
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        right_layout.addWidget(self.predict_btn)
        
        # Result display
        self.result_label = QLabel("No prediction yet")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; }")
        right_layout.addWidget(self.result_label)
        
        # Progress bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        right_layout.addWidget(self.confidence_bar)
        
        # Add stretch to push everything up
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)")
        
        if file_path:
            try:
                self.image_path = file_path
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.check_predict_ready()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model", "",
            "Model Files (*.h5);;All Files (*)")
        
        if file_path:
            try:
                # Register the custom layer
                custom_objects = {'SeqWeightedAttention': SeqWeightedAttention}
                self.model = tf.keras.models.load_model(file_path, custom_objects=custom_objects)
                QMessageBox.information(self, "Success", "Model loaded successfully!")
                self.check_predict_ready()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                self.model = None

    def check_predict_ready(self):
        self.predict_btn.setEnabled(
            self.model is not None and self.image_path is not None)

    def preprocess_image(self):
        try:
            # Read and resize image
            img = tf.keras.preprocessing.image.load_img(
                self.image_path,
                target_size=(self.height_spin.value(), self.width_spin.value())
            )
            # Convert to array and add batch dimension
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Normalize to [0,1]
            return img_array / 255.0
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict(self):
        try:
            # Preprocess image
            processed_image = self.preprocess_image()
            
            # Get prediction
            prediction = self.model.predict(processed_image)
            
            # Get probability (assuming binary classification)
            probability = float(prediction[0][0]) * 100
            threshold = self.threshold_slider.value()
            
            # Update UI
            if probability >= threshold:
                result_text = "Dog"
                result_probability = probability
            else:
                result_text = "Cat"
                result_probability = 100 - probability
            
            self.result_label.setText(f"{result_text} ({result_probability:.1f}%)")
            self.confidence_bar.setValue(int(result_probability))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Apply Fusion style with dark palette
    app.setStyle("Fusion")
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec()) 

if __name__ == "__main__":
    main()