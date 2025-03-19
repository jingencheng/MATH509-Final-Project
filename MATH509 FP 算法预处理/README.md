# Image Inpainting with Multiple Algorithms

A Python implementation of image inpainting using various machine learning algorithms including RBF, KNN, GMM, KMS, and MLP.

## Features

- Support for multiple inpainting algorithms:
  - Radial Basis Function (RBF)
  - K-Nearest Neighbors (KNN)
  - Gaussian Mixture Model (GMM)
  - K-Means (KMS)
  - Multi-Layer Perceptron (MLP)
- Automatic timing for performance measurement
- Configurable parameters for each algorithm
- Support for RGB image processing
- Patch-based inpainting approach

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Pillow
- scikit-learn

## Installation

- Install the required packages:

```bash
pip install numpy matplotlib pillow scikit-learn
```

## Usage

```python
from main import ImageInpainting
import numpy as np

# Initialize the inpainter with desired algorithm
inpainter = ImageInpainting(model_type='KNN')  # Options: 'RBF', 'KNN', 'GMM', 'KMS', 'MLP'

# Set parameters
image_name = "path/to/your/image.tif"
fill_rgb = np.array([255, 0, 0])  # Color to fill (RGB)
spacing = 3  # Sampling point spacing (1-9)
width = 2    # Sampling range width (1-2*spacing)
l2_coef = 0.5  # L2 regularization coefficient (for RBF)
tolerance = 0.5  # Color similarity threshold (0.1-0.5)
patch_size = 25  # Size of image patches

# Perform inpainting
result = inpainter.inpaint(
    image_name=image_name,
    fill_rgb=fill_rgb,
    spacing=spacing,
    width=width,
    l2_coef=l2_coef,
    tolerance=tolerance,
    patch_size=patch_size
)

# Access processing time
print(f"Processing time: {inpainter.time:.2f} seconds")
```

## Parameters

- `model_type`: Type of algorithm to use ('RBF', 'KNN', 'GMM', 'KMS', 'MLP')
- `fill_rgb`: RGB color values to fill (0-255)
- `spacing`: Distance between sampling points (1-9)
- `width`: Width of sampling range (1-2*spacing)
- `l2_coef`: L2 regularization coefficient (for RBF)
- `tolerance`: Color similarity threshold (0.1-0.5)
- `patch_size`: Size of image patches for processing

## Algorithm-Specific Parameters

- KNN: `n_neighbors` (default: 10)
- GMM: `n_components` (default: 3)
- KMS: `n_clusters` (default: 3)
- MLP: `hidden_layer_sizes` (default: (64, 64))

## Performance

The processing time for each algorithm can be accessed through the `time` attribute of the ImageInpainting instance after processing.
