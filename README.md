# Safety Helmet Classification for Thermal Energy Company
## Project Overview
This project leverages deep learning models to classify whether employees in thermal energy company monitoring images are wearing safety helmets. The goal is to enhance workplace safety by automating the identification of safety compliance. The project involves preprocessing raw images, training multiple state-of-the-art deep learning models, and comparing their performance.

## Key Features
- **Image Preprocessing**: Raw monitoring images are processed to improve training quality, including resizing, normalization, and data augmentation.
- **Deep Learning Models**: Trained multiple classification models, including:
  - LeNet
  - AlexNet
  - DenseNet
  - EfficientNet
  - ResNet
  - VGG
- **Model Evaluation**: Compared the performance of these models using key metrics like accuracy, precision, and recall.
- **Automated Monitoring**: Provides a scalable solution for real-time safety compliance monitoring.

## Dataset
- **Source**: Thermal energy company monitoring images.
- **Classes**:
  - `Helmet`: Employees wearing safety helmets.
  - `No Helmet`: Employees without safety helmets.
- **Size**: Approximately X images for training, Y images for validation, and Z images for testing.
- **Preprocessing Steps**:
  - Resizing images.
  - Normalizing pixel values.
  - Applying data augmentation techniques such as random rotation, flipping, and brightness adjustment.

## Models
### 1. **LeNet**
- Simple CNN architecture with fewer layers.
- Used as a baseline for performance comparison.

### 2. **AlexNet**
- Introduced deeper convolutional layers to improve feature extraction.

### 3. **DenseNet**
- Utilizes dense connections to improve gradient flow and reduce overfitting.

### 4. **EfficientNet**
- Balances network depth, width, and resolution for better performance with fewer parameters.

### 5. **ResNet**
- Incorporates residual connections to handle vanishing gradient problems in deep networks.

### 6. **VGG**
- Known for its simplicity and high performance in image classification tasks.

## Performance Comparison
| Model       | Accuracy | Precision | Recall |
|-------------|----------|-----------|--------|
| LeNet       | 92%      | 90%       | 90%    |
| AlexNet     | 96%      | 94%       | 94%    |
| DenseNet    | 97%      | 96%       | 97%    |
| EfficientNet| 98%      | 97%       | 98%    |
| ResNet      | 98%      | 98%       | 96%    |
| VGG         | 96%      | 95%       | 96%    |

## Setup Instructions
### Prerequisites
- Python 3.11
- GPU support
