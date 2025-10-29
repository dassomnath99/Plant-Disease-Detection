import os
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.shortcuts import render
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import json

from .models import PredictionHistory, DiseaseInfo

#Load model at startup
MODEL = None
CLASS_NAMES = []

def load_model():
    global MODEL, CLASS_NAMES
    try:
        MODEL = keras.models.load_model(settings.MODEL_PATH)
        # LOAD class names
        class_names_path = os.path.join(settings.BASE_DIR, 'models','class_names.json')
        with open(class_names_path, 'r') as f:
            CLASS_NAMES = json.load(f)
        print("Model and class names successfully loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_model()

def preprocess_image(image_file):
    try:
        img = Image.open(image_file)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))

        img_array = np.array(img)

        img_array = img_array.astype('float32') / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}") 

def get_fertilizer_recommendation(disease_name, plant_type):
    try:
        disease_info = DiseaseInfo.objects.filter(disease_name=disease_name).first()
        if disease_info:
            return {
                'fertilizer': disease_info.fertilizer,
                'description': disease_info.description,
                'prevention': disease_info.prevention,
                'treatment': disease_info.treatment
            }
    except:
        pass
    
    recommendations ={
        'healthy': {
            'fertilizer': 'Balanced NPK (10-10-10) fertilizer, apply every 2-3 weeks',
            'treatment': 'No treatment needed. Continue regular care.',
            'prevention': 'Maintain proper watering and sunlight',
            'description': 'Plant is healthy and showing no signs of disease'
        },
        'bacterial_spot': {
            'fertilizer': 'Copper-based fertilizer, reduce nitrogen',
            'treatment': 'Apply copper fungicide, remove infected leaves',
            'prevention': 'Avoid overhead watering, ensure good air circulation',
            'description': 'Bacterial infection causing spots on leaves'
        },
        'early_blight': {
            'fertilizer': 'High potassium fertilizer (5-10-15)',
            'treatment': 'Apply fungicide with chlorothalonil or copper',
            'prevention': 'Crop rotation, proper spacing, mulching',
            'description': 'Fungal disease causing dark spots with concentric rings'
        },
        'late_blight': {
            'fertilizer': 'Balanced fertilizer with micronutrients',
            'treatment': 'Remove infected parts immediately, apply fungicide',
            'prevention': 'Avoid wet foliage, ensure good drainage',
            'description': 'Serious fungal disease that can destroy crops quickly'
        },
        'leaf_mold': {
            'fertilizer': 'Potassium-rich fertilizer',
            'treatment': 'Improve ventilation, reduce humidity, apply fungicide',
            'prevention': 'Maintain low humidity, proper spacing',
            'description': 'Fungal disease thriving in high humidity'
        },
        'septoria_leaf_spot': {
            'fertilizer': 'Balanced NPK with calcium',
            'treatment': 'Remove infected leaves, apply fungicide',
            'prevention': 'Mulching, avoid splashing water on leaves',
            'description': 'Fungal disease causing small circular spots'
        },
        'spider_mites': {
            'fertilizer': 'Standard NPK, avoid excess nitrogen',
            'treatment': 'Apply neem oil or insecticidal soap',
            'prevention': 'Maintain humidity, regular inspection',
            'description': 'Pest infestation causing stippled leaves'
        },
        'target_spot': {
            'fertilizer': 'Balanced fertilizer with zinc',
            'treatment': 'Apply fungicide, improve air circulation',
            'prevention': 'Crop rotation, remove plant debris',
            'description': 'Fungal disease causing target-like lesions'
        },
        'mosaic_virus': {
            'fertilizer': 'Balanced NPK to support plant health',
            'treatment': 'No cure - remove infected plants to prevent spread',
            'prevention': 'Control aphids, use disease-resistant varieties',
            'description': 'Viral disease causing mottled, discolored leaves'
        },
        'yellow_leaf_curl': {
            'fertilizer': 'High nitrogen fertilizer to support growth',
            'treatment': 'Control whiteflies, remove infected plants',
            'prevention': 'Use insect screens, plant resistant varieties',
            'description': 'Viral disease transmitted by whiteflies'
        }
    }

    disease_lower = disease_name.lower()
    for key in recommendations:
        if key in disease_lower:
            return recommendations[key]
    return {
        'fertilizer': 'Balanced NPK fertilizer (10-10-10)',
        'treatment': 'Consult with agricultural expert for specific treatment',
        'prevention': 'Maintain good plant hygiene and growing conditions',
        'description': 'Disease detected - professional consultation recommended'
    }

@csrf_exempt
@require_http_methods(['POST'])
def predict_disease(request):
    pass