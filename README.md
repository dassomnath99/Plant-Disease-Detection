# 🌿 Custom CNN for Apple Leaf Disease Detection

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

> **Final Year Project | University of Kalyani (2025)**  
> **Accuracy: 93%** | **Classes: 3 Diseases + Healthy**  
> **Model: Custom 6-Block CNN with Batch Normalization & Dropout**

This repository contains the implementation of a **deep convolutional neural network (CNN)** designed to classify apple leaf diseases from RGB images. The model is trained on a dataset of over **32,000 labeled leaf images** and achieves **93% validation accuracy**, offering a scalable solution for early agricultural disease detection.

---

## 🎯 Objective

To develop a **robust, lightweight, and accurate** deep learning model that can:
- Distinguish between **healthy apple leaves** and those affected by **Apple Scab**, **Black Rot**, and **Cedar Rust**.
- Serve as the core inference engine for a **Django-based web application** used by farmers for real-time diagnosis.
- Reduce crop loss through **early, AI-powered intervention**.

---

## 🧠 Model Architecture

The model is a **custom-designed 6-block CNN** built with **TensorFlow/Keras**, featuring:

- **6 Convolutional Blocks**, each with:
  - Two `Conv2D` layers (ReLU activation, same padding)
  - `BatchNormalization` for stable training
  - `MaxPooling2D` (2×2) for spatial downsampling
- **Progressive channel depth**: 32 → 64 → 128 → 256 → 512 → 512
- **Flatten** layer followed by **three fully connected layers**:
  - Dense(1024) → Dropout(0.5)
  - Dense(512) → Dropout(0.5)
  - Dense(256) → Dropout(0.5)
- **Softmax output** for 4-class classification (3 diseases + healthy)

### Input & Output
- **Input shape**: `(256, 256, 3)` (RGB images)
- **Number of classes**: `4` (Note: code currently uses `num_classes = 3`; update if including "Healthy" as a class)
- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Metrics**: Accuracy

> 🔍 **Note**: The architecture avoids transfer learning to ensure full control over feature extraction and model interpretability.

---

## 🛠️ Key Features

- ✅ **Batch Normalization** after every convolutional layer for faster convergence  
- ✅ **Dropout (50%)** in dense layers to prevent overfitting  
- ✅ **ModelCheckpoint** callback to save the best model based on `val_loss`  
- ✅ GPU-accelerated training (auto-detected via `tf.config.list_physical_devices('GPU')`)  
- ✅ Designed for integration with **ImageDataGenerator** for on-the-fly augmentation

---

## ▶️ How to Use
```bash
# clone it
git clone https://github.com/dassomnath99/custom-cnn
# create a virtual environment
virtualenv .venv
# activate the virtual environment
.venv/Scripts/activate
# install the packages
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn jupyter
# start the jupyter notebook
jupyter notebook Custom_cnn.ipynb
