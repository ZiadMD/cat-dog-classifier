# Cat-Dog Image Classifier

A desktop application built with PySide6 (Qt) and TensorFlow for classifying images as either cats or dogs.

## Features

- Modern dark-themed UI using QDarkStyle
- Support for various image formats (PNG, JPG, JPEG, BMP, GIF)
- Adjustable input image dimensions
- Configurable classification threshold
- Real-time confidence visualization
- Custom attention mechanism support

## Requirements

```bash
pip install -r requirements.txt
```

## Dependencies

- Python 3.6+
- TensorFlow 2.x
- PySide6
- QDarkStyle
- keras-self-attention

## Usage

1. Run the application:
```bash
python app.py
```

2. Click "Load Model" to select your trained .h5 model file
3. Click "Load Image" to select an image for classification
4. Adjust the threshold if needed
5. Click "Predict" to classify the image

## Model Requirements

The application expects a TensorFlow model that:
- Takes image input of configurable dimensions
- Uses SeqWeightedAttention layer from keras-self-attention
- Outputs binary classification (cat/dog)

## License

MIT License 