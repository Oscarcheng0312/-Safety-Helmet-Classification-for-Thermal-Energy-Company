# Safety Helmet Classification for Thermal Energy Company

## Project Overview
This project leverages both traditional Python-based classification methods and deep learning models to classify whether employees in thermal energy company monitoring images are wearing safety helmets. The goal is to enhance workplace safety by automating safety compliance checks. By comparing traditional methods with deep learning models, the project highlights the performance improvements offered by state-of-the-art techniques.

## Key Features
- **Image Preprocessing**: Raw monitoring images are processed to improve classification quality, including resizing, normalization, and data augmentation.
- **Traditional Classification**: Implemented a K-Nearest Neighbors (KNN) algorithm as a baseline classification method.
- **Deep Learning Models**: Trained advanced classification models, including:
  - LeNet
  - AlexNet
  - DenseNet
  - EfficientNet
  - ResNet
  - VGG
- **Performance Comparison**: Compared traditional methods and deep learning models using metrics such as accuracy, precision, and recall.
- **Automated Monitoring**: Provides a scalable solution for real-time safety compliance monitoring.

## Dataset
- **Source**: Thermal energy company monitoring images.
- **Classes**:
  - `Helmet`: Employees wearing safety helmets.
  - `No Helmet`: Employees without safety helmets.
- **Size**: Approximately 5000 images for training, and 1000 images for testing.
- **Preprocessing Steps**:
  - Resizing images.
  - Normalizing pixel values.
  - Applying data augmentation techniques such as random rotation, flipping, and brightness adjustment.

## Methods
### 1. Traditional Python Classification: K-Nearest Neighbors (KNN)
- **Description**:
  - KNN is an instance-based classification algorithm that predicts the class of a sample based on the majority class of its `k` nearest neighbors.
- **Implementation Steps**:
  1. **Image Preprocessing**:
     - Images are converted to grayscale, resized to `28x28` pixels, and flattened into 1D feature vectors.
  2. **Classifier Training**:
     - Used `scikit-learn`'s `KNeighborsClassifier` with `k=5` neighbors.
     - Trained the classifier on the processed training dataset.
  3. **Performance Evaluation**:
     - Accuracy was calculated on the test dataset using `accuracy_score`.
  4. **Runtime and Resource Monitoring**:
     - Measured code execution time using the `time` module.
     - Monitored memory and CPU usage using `psutil`.
  5. **New Image Prediction**:
     - New images can be preprocessed and classified using the trained KNN model.
- **Results**:
  - Accuracy achieved with KNN: ~90%.

### 2. Deep Learning Models
- **Models Used**:
  - LeNet, AlexNet, DenseNet, EfficientNet, ResNet, VGG
- **Implementation**:
  - Built and trained models using PyTorch.
  - Conducted hyperparameter tuning to optimize performance.
- **Results**:
  - Accuracy ranged from 92% (LeNet) to 98% (EfficientNet and ResNet).

## Performance Comparison
| Model Type          | Model        | Accuracy | Precision | Recall |
|---------------------|--------------|----------|-----------|--------|
| **Traditional**     | KNN          | ~87%     | ~88%      | ~87%   |
| **Deep Learning**   | LeNet        | ~92%     | ~90%      | ~90%   |
|                     | AlexNet      | ~96%     | ~94%      | ~94%   |
|                     | DenseNet     | ~97%     | ~96%      | ~97%   |
|                     | EfficientNet | ~98%     | ~97%      | ~98%   |
|                     | ResNet       | ~98%     | ~98%      | ~96%   |
|                     | VGG          | ~96%     | ~95%      | ~96%   |

## Setup Instructions
### Prerequisites
- Python 3.11
- GPU support (optional but recommended for deep learning models)
  - **CUDA Toolkit**: Version compatible with GPU and PyTorch version.
  - **NVIDIA Drivers**: Ensure GPU drivers are up to date.
