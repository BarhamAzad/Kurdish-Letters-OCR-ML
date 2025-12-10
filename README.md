# Kurdish Full Alphabet OCR - Advanced CNN Classification Model

A PyTorch-based Convolutional Neural Network for classifying the entire Kurdish handwritten alphabet  using deep learning and advanced techniques.



## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Process](#training-process)
- [Model Performance](#model-performance)
- [Results](#results)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

This project trains and deploys an advanced CNN model to recognize and classify the full Kurdish alphabet with **exceptional performance**:
- Full kurdish alphabet (33 letters)
- **Achieves 95.87% validation accuracy** on unseen test data
- Optimized for fast inference (real-time performance)
- Built with cutting-edge deep learning techniques
- Comprehensive dataset of 33,067 images across 33 classes

The model processes 64x64 grayscale images using state-of-the-art techniques including residual connections, data augmentation, and mixed precision training.

## Features

### 🧠 Advanced Neural Architecture
- **PyTorch-based Advanced CNN architecture** with residual connections (ResNet-style)
- **1.49M trainable parameters** optimized for efficiency
- **LeakyReLU activation** (0.1 slope) for improved gradient flow
- **Global Average Pooling** for flexible input sizes
- **Projection shortcuts** to handle channel dimension changes

### 🚀 Performance Optimized
- **GPU support** with automatic CPU fallback
- **Mixed Precision Training (AMP)** for faster computation
- **Real-time inference** (<20ms on GPU, <50ms on CPU)
- **95.87% validation accuracy** achieved in just 15.8 minutes training time

### 📊 Comprehensive Dataset
- **33,067 total images** across 33 Kurdish character classes
- **Consistent distribution** (1,008 images per class for most letters)
- **Real-world handwritten samples** for authentic training data
- **Proper Unicode mapping** for each Kurdish character

### 🛠️ Robust Tools
- **Batch image testing** capabilities
- **Detailed confidence scores** and probability visualization
- **Live camera inference** with interactive GUI
- **Modular, well-documented code**
- **Cross-platform compatibility**

### 🎯 Deep Learning Features
- **Advanced data augmentation** (rotation, scaling, noise, elastic transforms)
- **Early stopping** to prevent overfitting
- **Learning rate scheduling** for optimal convergence
- **Batch normalization** for training stability
- **Dropout regularization** (30%) to prevent overfitting

## Architecture

### KurdishAdvancedCNN Architecture

The KurdishAdvancedCNN is a sophisticated 4-layer convolutional neural network with residual connections inspired by ResNet architecture:

```
Input (3x64x64) [Grayscale images converted to 3-channel for augmentation compatibility]

├── Block 1: Input → 32 channels
│   ├── Conv2d(3→32, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Conv2d(32→32, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Add + LeakyReLU(0.1) + MaxPool(2x2) + Dropout2D(0.3)
│   └── Residual connection (direct path)

├── Block 2: 32 → 64 channels
│   ├── Projection Layer: 32→64 (1x1 Conv + BatchNorm for dimension matching)
│   ├── Conv2d(32→64, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Conv2d(64→64, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Add + LeakyReLU(0.1) + MaxPool(2x2) + Dropout2D(0.3)
│   └── Residual connection (projection + original path)

├── Block 3: 64 → 128 channels
│   ├── Projection Layer: 64→128 (1x1 Conv + BatchNorm)
│   ├── Conv2d(64→128, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Conv2d(128→128, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Add + LeakyReLU(0.1) + MaxPool(2x2) + Dropout2D(0.3)
│   └── Residual connection (projection + original path)

├── Block 4: 128 → 256 channels
│   ├── Projection Layer: 128→256 (1x1 Conv + BatchNorm)
│   ├── Conv2d(128→256, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Conv2d(256→256, 3x3) + BatchNorm + LeakyReLU(0.1)
│   ├── Add + LeakyReLU(0.1) + MaxPool(2x2) + Dropout2D(0.3)
│   └── Residual connection (projection + original path)

├── Global Average Pooling
├── Flatten (256 → 256)
├── FC(256→512) + BatchNorm1D + LeakyReLU(0.1) + Dropout(0.3)
├── FC(512→256) + BatchNorm1D + LeakyReLU(0.1) + Dropout(0.3)
└── FC(256→33 classes) + Softmax

Total Parameters: 1,491,009 trainable weights
```

### Key Architectural Features
- **Residual Connections**: Enable training of deep networks without vanishing gradients
- **Projection Shortcuts**: Handle channel dimension mismatches between residual blocks
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **LeakyReLU Activation**: Prevents dead neurons and improves gradient flow
- **Dropout Regularization**: Prevents overfitting with 30% dropout rate
- **Global Average Pooling**: Reduces parameters and prevents overfitting

### Data Augmentation Pipeline
- **Horizontal Flip**: 30% probability
- **Rotation**: ±15 degrees with 30% probability
- **Affine Transforms**: Translation, scaling, rotation with 30% probability
- **Brightness/Contrast**: ±20% limits with 30% probability
- **Gaussian Noise**: With 20% probability
- **Elastic Transform**: With 10% probability
- **Normalization**: ImageNet statistics (mean=[0.456], std=[0.224])

## Dataset

### Dataset Composition

The dataset contains comprehensive Kurdish character data with detailed image counts:

| Character | Name | Unicode | Image Count | Description |
|-----------|------|---------|-------------|-------------|
| **س** | Seen | U+0633 | **1,071** | Highest count character |
| **ک** | Kaf (keheh) | U+06A9 | 1,025 | Arabic letter keheh |
| **ا** | Alef | U+0627 | 1,008 | Basic vowel letter |
| **ب** | Be | U+0628 | 1,008 | Basic consonant |
| **ت** | Te | U+062A | 1,008 | Consonant with three dots |
| **ج** | Jim | U+062C | 1,008 | Consonant with distinctive shape |
| **ح** | Hah | U+062D | 1,008 | Consonant with bowl shape |
| **خ** | Khe | U+062E | 1,008 | Consonant with vertical stem |
| **د** | Dal | U+062F | 1,008 | Vowel consonant |
| **ر** | Re | U+0631 | 1,008 | Consonant with curved shape |
| **ز** | Ze | U+0632 | 1,008 | Similar to Re |
| **ش** | Sheen | U+0634 | 1,008 | Visually similar to Seen |
| **ع** | Ayn | U+0639 | 1,008 | Complex joining character |
| **غ** | Ghayn | U+063A | 1,008 | Similar to Ayn |
| **ف** | Feh | U+0641 | 1,008 | Consonant with dot placement |
| **ق** | Qaf | U+0642 | 1,008 | Consonant with bowl shape |
| **ك** | Kaf | U+0643 | 1,008 | Standard Arabic Kaf |
| **ل** | Lam | U+0644 | 1,008 | Consonant that connects |
| **م** | Mim | U+0645 | 1,008 | Closed bowl shaped |
| **ن** | Nun | U+0646 | 1,008 | Consonant with distinctive tail |
| **ه** | Heh | U+0647 | 1,008 | Final form separator |
| **و** | Waw | U+0648 | 1,008 | Round vowel character |
| **وو** | Ve | U+06CB | 1,008 | Kurdish-specific double Waw |
| **پ** | Pe | U+067E | 1,008 | Consonant with two dots |
| **چ** | Che | U+0686 | 1,008 | Visually similar to Jim |
| **ڕ** | Re with ring | U+0695 | 1,008 | Distinctive Kurdish with ring |
| **ژ** | Zhe | U+0698 | 1,008 | Distinctive Kurdish with three dots |
| **ڤ** | Pe with three dots | U+06A4 | 1,008 | Kurdish-specific with three dots |
| **گ** | Gaf | U+06AF | 1,008 | Similar to Kaf |
| **ڵ** | Lam with line | U+06B5 | 1,008 | Kurdish-specific Lam variant |
| **ھ** | Heh goal | U+06BE | 1,008 | Kurdish-specific Heh variant |
| **ۆ** | OE | U+06C6 | 1,008 | Kurdish-specific vowel |
| **ی** | Farsi Yeh | U+06CC | **731** | Lowest count character |

**Total Dataset Statistics:**
- **Total Images**: 33,067
- **Total Classes**: 33 Kurdish letters
- **Training Samples**: 26,453 (80%)
- **Validation Samples**: 6,614 (20%)
- **Average per Class**: 1,008 images
- **Image Size**: 64x64 pixels (resized from original)
- **Format**: JPG, PNG, JPEG, BMP

### Data Distribution Analysis
- **Balanced dataset**: Most classes have exactly 1,008 images
- **Highest count**: **س** (Seen) with 1,071 images
- **Lowest count**: **ی** (Farsi Yeh) with 731 images
- **Kurdish-specific letters**: **ڕ**, **ژ**, **ڤ**, **ڵ**, **ۆ**, **وو** are distinctive to Kurdish
- **Arabic script base**: Most letters follow Arabic script with Kurdish modifications

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- 8GB+ RAM recommended

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/kurdish-ocr.git
cd kurdish-ocr
```

2. **Create virtual environment** (recommended):
```bash
python -m venv kurdish_ocr_env
source kurdish_ocr_env/bin/activate  # On Windows: kurdish_ocr_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install opencv-python numpy torch torchvision scikit-learn albumentations tqdm pillow ipywidgets matplotlib seaborn
```

### Detailed Dependencies List
- **PyTorch 1.9+**: Deep learning framework
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning utilities
- **albumentations**: Advanced image augmentation
- **tqdm**: Progress bars
- **Pillow**: Image loading alternative
- **matplotlib/seaborn**: Visualization
- **ipywidgets**: Interactive notebook widgets

## Quick Start

### Option 1: Use Pre-trained Model (Recommended for Testing)

1. **Download pre-trained model** (if not already in repository):
```bash
# Model should already be included as kurdish_letter_model_pytorch.pth
```

2. **Test single image**:
```bash
jupyter notebook testing_CNN.ipynb
```
Then run the prediction code:
```python
image_path = "path/to/your/kurdish_letter_image.jpg"
predicted_class, confidence, probabilities = predict_single_image(image_path)
print(f"Predicted: {predicted_class} with {confidence:.2f}% confidence")
```

### Option 2: Train the Model from Scratch

1. **Prepare dataset**:
```bash
python3 create_combined_dataset.py
```

2. **Start training**:
```bash
jupyter notebook training_CNN.ipynb
```

3. **Monitor training progress** in real-time as the model achieves:
   - Epoch 1: ~21% accuracy
   - Epoch 5: ~88% accuracy
   - Epoch 10: ~93% accuracy
   - Best: 95.87% accuracy at epoch 37

## Training Process

### Training Configuration
```python
IMG_SIZE = 64                    # Input image dimensions (64x64 pixels)
CHANNELS = 1 (converted to 3)    # Single channel converted for augmentation
EPOCHS = 100                     # Maximum training epochs
BATCH_SIZE = 32                  # Images per batch
LEARNING_RATE = 0.001            # Adam optimizer learning rate
EARLY_STOPPING_PATIENCE = 15     # Early stopping patience
DROPOUT_RATE = 0.3               # Dropout regularization rate
```

### Training Dynamics
- **Optimizer**: Adam with weight decay (1e-4)
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Monitoring**: Generalization monitor to detect overfitting
- **Checkpointing**: Models saved only when validation accuracy improves

### Training Results
- **Best Epoch**: 37
- **Best Validation Accuracy**: **95.87%**
- **Final Training Accuracy**: 94.6%
- **Training Time**: 15.8 minutes on GPU
- **Total Parameters**: 1,491,009
- **Early Stopping**: Triggered at epoch 52

### Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| Best Validation Accuracy | **95.87%** | Achieved at epoch 37 |
| Final Training Accuracy | 94.6% | At stopping epoch 52 |
| Best Validation Loss | 0.1271 | At epoch 38 |
| Final Training Loss | 0.1792 | At epoch 52 |
| Training Time | 15.8 minutes | Total training duration |
| Total Parameters | 1.49M | Trainable weights |

## Model Performance

### Training Dynamics Analysis
- **Epoch 1**: 21.23% validation accuracy (rapid initial learning)
- **Epoch 5**: 88.13% validation accuracy (rapid convergence)
- **Epoch 10**: 92.88% validation accuracy (continued improvement)
- **Epoch 37**: 95.87% validation accuracy (best performance)
- **Final**: Maintained high performance with minimal overfitting

### Key Performance Indicators
1. **Excellent Generalization**: Validation accuracy consistently higher than training accuracy due to data augmentation
2. **Rapid Convergence**: Achieved >90% accuracy within first 10 epochs
3. **Minimal Overfitting**: Training and validation curves remain close throughout training
4. **Stable Training**: No loss spikes or performance degradation

### Computational Performance
- **Inference Speed**: <20ms per image on GPU
- **Inference Speed**: <50ms per image on CPU
- **Model Size**: ~17.2 MB (portable for edge deployment)
- **Memory Usage**: Optimized with mixed precision training

## Results

### Comprehensive Performance Analysis

The model achieved remarkable results across all evaluation metrics:

#### Accuracy Performance
- **Best Validation Accuracy**: 95.87%
- **Training Accuracy**: 94.6%
- **Accuracy Gap**: 1.27% (indicating excellent generalization)

#### Training Efficiency
- **Convergence Speed**: 88.13% accuracy by epoch 5
- **Training Time**: 15.8 minutes on GPU
- **Memory Efficiency**: Optimized with mixed precision
- **Resource Utilization**: Balanced computation and accuracy

#### Dataset Performance
The model performs consistently across all 33 Kurdish character classes:
- **Balanced Performance**: Consistent accuracy across all characters
- **Character Recognition**: All 33 classes achieve high recognition rates
- **Visual Similarity Handling**: Successfully distinguishes visually similar characters

### Real-world Application Potential
- **Digitization Projects**: Suitable for Kurdish document digitization
- **Educational Tools**: Kurdish language learning applications
- **Cultural Preservation**: Digital preservation of Kurdish literature
- **Government Applications**: Processing Kurdish administrative documents

## Usage Examples

### Basic Single Image Prediction
```python
import torch
import cv2
import numpy as np

# Load test image
image_path = "test_kurdish_letter.jpg"

# Predict using trained model
predicted_class, confidence, probabilities = predict_single_image(image_path)

# Display results
print(f"Predicted Kurdish character: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
print(f"Full probability distribution: {probabilities}")
```

### Batch Image Processing
```python
# Test multiple images from directory
test_multiple_images(folder_path="path/to/test/images/")

# Test specific files
test_multiple_images(file_list=["letter1.jpg", "letter2.png", "letter3.jpg"])
```

### Live Camera Interface
The testing notebook includes an interactive GUI:
1. **Camera Selection**: Choose from multiple camera sources
2. **ROI Capture**: Define region of interest for character recognition
3. **Real-time Processing**: Instant prediction and feedback
4. **Confidence Display**: Shows confidence percentage and alternatives

### Custom Implementation
```python
# Load and use model programmatically
model = KurdishAdvancedCNN(num_classes=33)
model.load_state_dict(torch.load('kurdish_letter_model_pytorch.pth', map_location=device)['model_state_dict'])
model.eval()

# Process custom image
def process_custom_image(image_path):
    # Preprocessing (same as training)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    image = np.stack([image] * 3, axis=-1)  # Convert to 3 channels
    image = image.astype('float32') / 255.0
    image = (image - 0.456) / 0.224  # Normalize with ImageNet stats
    image = np.transpose(image, (2, 0, 1))
    img_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return predicted_idx.item(), confidence.item()
```

## Technical Details

### Advanced Deep Learning Techniques

#### 1. Residual Connections with Projection
- **Purpose**: Prevent vanishing gradients in deep networks
- **Implementation**: 1x1 Convolutional projection layers for dimension matching
- **Benefit**: Enables training of 4-layer deep network effectively

#### 2. Mixed Precision Training (AMP)
- **Purpose**: Accelerate training while maintaining accuracy
- **Implementation**: Automatic Mixed Precision using GradScaler
- **Benefit**: 2-3x faster training on GPU with same accuracy

#### 3. Data Augmentation Strategy
- **Horizontal Flip**: 30% probability for orientation invariance
- **Rotation**: ±15 degrees for angle invariance
- **Affine Transforms**: Translation, scaling for position invariance
- **Photometric**: Brightness, contrast adjustments for lighting invariance
- **Noise**: Gaussian noise for robustness
- **Elastic**: Elastic transforms for handwriting variations

#### 4. Regularization Techniques
- **Batch Normalization**: After each convolutional layer
- **Dropout**: 30% in convolutional and fully connected layers
- **Early Stopping**: Prevents overfitting after 15 epochs without improvement
- **Weight Decay**: Adam optimizer L2 regularization (1e-4)

### Model Architecture Details

#### Convolutional Blocks
Each residual block contains:
- Two Conv2d layers (3x3 kernels)
- Batch Normalization after each Conv layer
- LeakyReLU activation (0.1 slope)
- MaxPooling (2x2) for spatial reduction
- Dropout for regularization

#### Classification Head
- Global Average Pooling (replaces Flatten)
- Three Fully Connected layers with BatchNorm and Dropout
- Softmax for multi-class probability distribution

### Hardware Optimization
- **GPU Support**: Automatic detection and utilization
- **CPU Fallback**: Seamless operation without GPU
- **Memory Management**: Optimized for standard hardware
- **Batch Processing**: Efficient GPU utilization during inference


## Notebooks Overview

### training_CNN.ipynb

**Primary Purpose**: Comprehensive training pipeline for Kurdish character recognition

**Key Sections**:

1. **Environment Setup**
   - Library imports (PyTorch, OpenCV, scikit-learn, NumPy, albumentations)
   - Device detection (GPU/CPU)
   - Configuration parameters

2. **Advanced Model Architecture**
   - KurdishAdvancedCNN class definition with residual connections
   - Projection shortcuts for channel dimension matching
   - LeakyReLU activation and batch normalization
   - Global Average Pooling implementation

3. **Custom Dataset Class**
   - KurdishAlphabetDataset with automated folder scanning
   - Image loading and preprocessing pipeline
   - Label encoding and validation data splitting
   - Data augmentation for training data

4. **Data Preparation Pipeline**
   - Load 33,067 images across 33 classes
   - 80:20 train-validation split (26,453/6,614 samples)
   - Apply albumentations augmentation pipeline
   - Normalize with ImageNet statistics

5. **Optimized Training Loop**
   - Mixed Precision Training with GradScaler
   - Generalization Monitor for overfitting detection
   - Learning Rate Scheduling with ReduceLROnPlateau
   - Checkpoint saving for best validation accuracy
   - Real-time progress tracking with tqdm

6. **Training Results & Logging**
   - Detailed epoch-by-epoch metrics
   - Training/Validation loss and accuracy curves
   - Early stopping implementation
   - Model checkpointing

7. **Comprehensive Evaluation**
   - `evaluate_model_comprehensive()` function
   - Per-class accuracy analysis
   - Confusion matrix heatmap
   - Top-K accuracy calculation
   - Error analysis and visualization

### testing_CNN.ipynb

**Primary Purpose**: Inference and evaluation pipeline for trained Kurdish OCR model

**Key Sections**:

1. **Model Loading & Setup**
   - Load trained KurdishAdvancedCNN architecture
   - Load pre-trained weights from checkpoint
   - Load label encoder for character mapping
   - Device configuration (GPU/CPU auto-detect)

2. **Prediction Functions**
   - `predict_single_image()`: Single image classification
   - `predict_from_array()`: Array-based prediction
   - Confidence scoring and probability distribution
   - Error handling and validation

3. **Interactive GUI Interface**
   - Live camera capture with ROI selection
   - Real-time character recognition
   - Confidence visualization
   - Camera index selection
   - Start/Stop controls

4. **Batch Processing Tools**
   - `test_multiple_images()` function
   - Folder-based testing
   - File list testing
   - Confidence analysis
   - Error visualization
