import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

CONFIG = {
    'dataset_path': 'data/plant_disease_data',  
    'image_size': 224,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
    'model_save_path': 'models/plant_disease_model.h5',
    'class_names_path': 'models/class_names.json',
    'history_plot_path': 'models/training_history.png',
    'confusion_matrix_path': 'models/confusion_matrix.png',
}

def prepare_dataset(dataset_path, image_size, batch_size, validation_split):
    print("Scanning dataset directory...")
    class_names = sorted([d for d in os.listdir(dataset_path) if   os.path.isdir(os.path.join(dataset_path,d))])

    num_classes = len(class_names)
    print(f"✅ Found {num_classes} disease classes: ")
    for i, cls in enumerate(class_names):
        img_count = len(os.listdir(os.path.join(dataset_path,cls)))
        print(f"  {i+1}.{cls}: {img_count} images")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for testing (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    print("\n📊 Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    
    # Load validation data
    print("📊 Loading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    
    return train_generator, validation_generator, class_names


# BUILD CNN MODEL
def build_model(num_classes, input_size=224):
    print("\n 🏗️ Building model architecture...")
    print("   Using Transfer Learning with MobileNetV2...")
    base_model = keras.applications.MobileNetV2(
        input_shape=(input_size,input_size,3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers
    base_model.trainable = False

    #Build complete model
    model = models.Sequential([
        keras.Input(shape=(input_size,input_size,3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    print("✅ Model architecture: ")
    model.summary()

    return model

def build_custom_cnn(num_classes, input_size=224):
    print("\n🏗️ Building custom CNN Architecture...")
    model = models.Sequential([
        #Block 1
        keras.Input(shape=(input_size, input_size, 3)),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        #Block 2
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        #Block 3
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        #Block 4
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        #Dense Layers
        layers.Flatten(),
        layers.Dense(512,activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256,activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')

    ])

    print("✅ Custom CNN architecture:")
    model.summary()

    return model

# Compile Model
def compile_model(model, learning_rate=0.001):
    print("\n⚙️ Compiling model...")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',keras.metrics.Precision(),keras.metrics.Recall()]
    )

    print("✅ Model compiled successfully")
    return model

# callbacks

def get_callbacks(model_path):
    callbacks = [
        #save best model
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only = True,
            mode='max',
            verbose=1
        ),
        #early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        #learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
    ]
    return callbacks

def train_model(model, train_generator, validation_generator, epochs, callbacks):
    # Train the Model
    print("\n 🚀 Starting Training...")
    print(f"  Epochs:{epochs}")
    print(f"  Batch Size: {train_generator.batch_size}")
    print(f"  Training Samples: {train_generator.samples}")
    print(f"  Validation Samples: {validation_generator.samples}")

    history = model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = epochs,
        callbacks = callbacks,
        verbose=1
    )

    print("\n✅ Training Completed!")
    return history

# Evaluate Model
def evaluate_model(model, validation_generator):
    print("\n 📊 Evaluation model on validation set...")

    results = model.evaluate(validation_generator, verbose=1)

    print(f"\n  Validation Loss: {results[0]:.4f}")
    print(f"  Validation Accuracy: {results[1]:.4f}")
    print(f"  Validation Precision: {results[2]:.4f}")
    print(f"  Validation Recall: {results[3]:.4f}")

    return results

# Save model

def save_model(model, save_path):
    # save model in h5 format
    print(f"\n 🛟 Saving Model to {save_path}...")

    dirpath = os.path.dirname(save_path) or '.'
    os.makedirs(dirpath, exist_ok=True)

    model.save(save_path)

    print(f"✅ Model Saved successfully!")

def save_class_names(class_names, save_path):
    print(f"\n🛟 Saving class names to {save_path}")
    dirpath = os.path.dirname(save_path) or '.'
    if os.path.exists(dirpath):
        if not os.path.isdir(dirpath):
            print(f"⚠️ Path '{dirpath}' exists and is not a directory; saving class names to current working directory instead.")
            dirpath = '.'
            save_path = os.path.join(dirpath, os.path.basename(save_path))
    else:
        os.makedirs(dirpath, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(class_names, f, indent=4)

    print("✅ Class names saved successfully!")

def save_model_tflite(model, save_path):
    print(f"\n💾 Converting model to TensorFlow Lite format...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()

    tflite_save_path = save_path.replace('.h5','.tflite')
    os.makedirs(os.path.dirname(tflite_save_path), exist_ok=True)

    with open(tflite_save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ TFLite model saved to {tflite_save_path}")
    print(f"   File size: {os.path.getsize(tflite_save_path) / (1024*1024):.2f} MB")

# Plotting and Visualition
def plot_training_history(history, save_path):
    print(f"\n 📈 Plotting training history...")
    fig, axes = plt.subplots(1,2, figsize=(15,5))

    #Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training history plot saved to {save_path}")
    
    plt.close()

# loading and test saved model

def load_and_test_model(model_path, class_names_path, test_image_path):
    print(f"\n⚙️ Loading saved model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model Loaded Successfully!")

    with open(class_names_path,'r') as f:
        class_names = json.load(f)

    img = keras.preprocessing.image.load_img(test_image_path, target_size=224)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    img_array /= 255.0
    img_array /= 255.0

    # make predictions
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    predicted_disease = class_names[predicted_class_idx]

    print(f"\n🔍  Prediction Results: ")
    print(f"  Disease: {predicted_disease}")
    print(f"  Confidence: {confidence*100:.2f}%")

    #show top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    print(f"\n Top 3 Predictions: ")
    for i, idx in enumerate(top_3_indices):
        print(f"  {i+1}.{class_names[idx]}: {predictions[0][idx]*100:.2f}%")

# main execution

def main():
    # main training execution

    print("+"*60)
    print(" 🌿 PLANT DISEASE DETECTION MODEL TRAINING")
    print("+"*60)

    CONFIG = {
        'dataset_path': 'data/plant_disease_data/',
        'image_size': 224,
        'batch_size': 8,
        'validation_split': 0.2,
        'test_split': 0.1,
        'epochs': 5,
        'learning_rate': 0.001,
        'model_save_path': 'models/plant_disease_model.h5',
        'class_names_path': 'models/class_names.json',
        'history_plot_path': 'models/training_history.png',
        'confusion_matrix_path': 'models/confusion_matrix.png',
    }

    # dataset preparation
    train_generator, validation_generator, class_names = prepare_dataset(
        CONFIG['dataset_path'],
        CONFIG['image_size'],
        CONFIG['batch_size'],
        CONFIG['validation_split']
    )

    # BUILD THE MODEL
    model = build_model(len(class_names), CONFIG['image_size'])

    # COMPILE MODEL
    model = compile_model(model, CONFIG['learning_rate'])

    # Get callbacks
    callbacks = get_callbacks(CONFIG['model_save_path'])

    # Train model
    history = train_model(
        model,
        train_generator,
        validation_generator,
        CONFIG['epochs'],
        callbacks
    )

    #Evaluate model
    evaluate_model(model, validation_generator)
    # Save Model
    save_model(model, CONFIG['model_save_path'])
    save_class_names(class_names, CONFIG['class_names_path'])

    # Save TFlite version
    save_model_tflite(model, CONFIG['model_save_path'])

    # Plot training history
    plot_training_history(history, CONFIG['history_plot_path'])

    #test loading saved model
    test_image_path = "data/plant_disease_data/test/Tomato___Late_blight/0a1f6e5b-7f02-4f12-bc10-6d3f1f2f3b4c___RS_Late.B 3536.JPG"
    load_and_test_model(
        CONFIG['model_save_path'],
        CONFIG['class_names_path'],
        test_image_path
    )
    print("\n"+"+" * 60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model Files:")
    print(f"  - {CONFIG['model_save_path']}")
    print(f"  - {CONFIG['model_save_path'].replace('.h5','.tflite')}")
    print(f"  - {CONFIG['class_names_path']}")
    print(f"  - {CONFIG['history_plot_path']}")

if __name__ == "__main__":
    main()