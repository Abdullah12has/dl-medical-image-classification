
# Deep Learning for Medical Image Classification


## You will find all the submission materials here in the repo including the visualization results and also the pth files for the models we trained.


## Overview

This repository implements deep learning techniques to classify medical images, particularly diabetic retinopathy detection. The project leverages transfer learning, advanced preprocessing methods, and ensemble learning techniques to enhance model performance and interpretability.



---

## Repository Structure

- **`bagging/`**  
  Contains scripts for bagging ensemble learning with ResNet18, ResNet34, and VGG16.

- **`boosting/`**  
  Implements boosting methods for the same models.

- **`stacking/`**  
  Scripts for stacking ensemble methods applied to the selected models.

- **`aio.py`**  
  A configurable all-in-one script that integrates preprocessing methods, ensemble techniques, and model training settings.

- **`visualizations/`**  
  Includes tools and results for visualizations such as training/validation loss graphs and GradCAM-based explainable AI outputs.

- **`trained_models/`**  
  This folder contains the trained models for ResNet18, ResNet34, and VGG16. Models are stored based on the ensemble technique and preprocessing configurations.

---

## How to Run

### Prerequisites
- Python 3.10+
- Install the required dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Code

#### Using All-in-One Configuration (`aio.py`)
1. **Run the Main Script**:  
   Execute the `aio.py` file to perform the tasks based on pre-defined configurations:  
   ```bash
   python aio.py
   ```
2. **Customize Configurations**:  
   Update `aio.py` for specific preprocessing techniques, ensemble methods, or model choices.

#### Running Specific Ensemble Techniques
1. **Navigate to the Folder**:  
   Go to the folder corresponding to the desired ensemble technique (`bagging/`, `boosting/`, or `stacking/`).

2. **Run Model-Specific Files**:  
   Each folder contains separate scripts for ResNet18, ResNet34, and VGG16. Execute the appropriate file.  
   Example for bagging with ResNet18:  
   ```bash
   cd bagging
   python resnet18Bagging.py
   ```
   Replace `resnet18Bagging.py` with `resnet34Bagging.py` or `vgg16Bagging.py` for other models.

---

## Features

### 1. Transfer Learning
Fine-tuned the following pre-trained models:
- ResNet18
- ResNet34
- VGG16

Trained on the DeepDRiD dataset to improve diabetic retinopathy detection.

### 2. Image Preprocessing
Explored techniques for enhancing input data:
- **Ben Graham Preprocessing**
- **Circle Cropping**
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Gaussian Blur** and **Sharpening**

### 3. Ensemble Learning
Implemented the following ensemble methods:
- Bagging
- Boosting
- Stacking

### 4. Explainable AI and Visualizations
- **GradCAM**: Highlights influential regions in medical images for decision-making.
- **Graphs**: Tracks training/validation accuracy and loss over epochs.

---

## Results

- **Metrics**: Evaluated using Cohen’s Kappa score, accuracy, and loss.
- **Ensemble Performance**: Demonstrated notable improvements compared to single models.
- **Visual Insights**: GradCAM effectively identified retinal regions critical for classification.

