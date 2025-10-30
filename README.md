Plant Disease Detection

A Django web application that uses an advanced custom CNN model to detect plant diseases from leaf images.
🚀 Features

    Advanced CNN Architecture with multiple convolutional blocks

    Batch Normalization for stable training

    Dropout Layers to prevent overfitting

    Real-time disease prediction

    User-friendly web interface

📁 Project Structure

plant-disease/
├── model_training/
│   ├── train_model.py      # Advanced CNN training
│   └── requirements.txt
├── disease_detection/      # Django app

🧠 Enhanced CNN Architecture
Model Features:

    - 4 Convolutional Blocks with increasing complexity

    - Batch Normalization after each conv layer

    - Progressive Dropout (0.25 → 0.5)

    - Dense Layers with 512 and 256 units

    - Softmax Output for multi-class classification

Block Structure:

Input (224x224x3)
├── Conv2D (64) → BatchNorm → Conv2D (64) → BatchNorm → MaxPool → Dropout
├── Conv2D (128) → BatchNorm → Conv2D (128) → BatchNorm → MaxPool → Dropout
├── Conv2D (256) → BatchNorm → Conv2D (256) → BatchNorm → MaxPool → Dropout
├── Conv2D (512) → BatchNorm → Conv2D (512) → BatchNorm → MaxPool → Dropout
└── Dense(512) → Dense(256) → Output(num_classes)


⚡ Quick Start
1. Train Enhanced CNN Model
cd model_training
pip install -r requirements.txt
python train_model.py

2. Launch Django App
cd ../disease_detection
python manage.py runserver

🛠️ Tech Stack

    - Deep Learning: TensorFlow/Keras with Custom CNN

    - Backend: Django

    - Frontend: Bootstrap, JavaScript

    - Image Processing: OpenCV, Pillow

📊 Model Specifications

    - Input: 224x224 RGB images

    - Layers: 4 Conv blocks + 2 Dense layers

    - Regularization: BatchNorm + Dropout

    - Activation: ReLU + Softmax

    - Classes: 38 plant diseases

🔧 Requirements

tensorflow>=2.8
django>=3.2
pillow
numpy
opencv-python

🌐 Usage

    - Upload leaf image

    - Advanced CNN processes the image

    - Get instant disease prediction