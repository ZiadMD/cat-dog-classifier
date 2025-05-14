#!/usr/bin/env python3
import os
import sys
import tensorflow as tf
from pathlib import Path

def convert_h5_to_keras(input_path: str, output_path: str = None) -> str:
    """
    Convert a .h5 model file to .keras format
    
    Args:
        input_path: Path to the .h5 model file
        output_path: Optional path for the output .keras file. If not provided,
                    will use the same name as input but with .keras extension
    
    Returns:
        Path to the converted .keras file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model file not found: {input_path}")
        
    if not input_path.endswith('.h5'):
        raise ValueError("Input file must be a .h5 file")
    
    # Load the model
    model = tf.keras.models.load_model(input_path)
    
    # If no output path specified, create one based on input path
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.keras'))
    
    # Save in .keras format
    model.save(output_path)
    print(f"Model converted successfully!")
    print(f"Original model: {input_path}")
    print(f"Converted model: {output_path}")
    
    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_model.py <path_to_h5_model> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_h5_to_keras(input_path, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 