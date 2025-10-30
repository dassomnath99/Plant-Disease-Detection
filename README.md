# Plant Disease Detection

A compact deep-learning project to detect and classify diseases in plant leaves from images. The repository contains data preparation, model training, evaluation, and inference utilities suited for quick experiments and reproducible results.

## Features
- Image preprocessing and augmentation pipelines
- CNN-based training and transfer-learning support (e.g., ResNet, EfficientNet)
- Training, validation, and test evaluation scripts
- Single-image inference CLI
- Exportable model for deployment

## Requirements
- Python 3.9+
- GPU recommended for training
- Key Python packages: numpy, pandas, torch (or tensorflow), torchvision (or tf.keras), Pillow, scikit-learn, matplotlib

Install:
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```


## Quickstart — Training
Train with sensible defaults; scripts accept config/CLI args:
```bash
python train.py --data ./dataset --model resnet50 --epochs 30 --batch-size 32 --lr 1e-4 --output ./models
```
Outputs:
- Best model checkpoint (./models)
- Training logs and metrics (tensorboard or CSV)

## Evaluation
Evaluate a saved checkpoint on the test set:
```bash
python evaluate.py --data ./dataset/test --checkpoint ./models/best.pth --batch-size 32
```
Produces:
- Accuracy, precision, recall, F1 per class
- Confusion matrix and sample prediction plots

## Inference
Predict a single image:
```bash
python predict.py --image ./examples/leaf.jpg --checkpoint ./models/best.pth --labels labels.txt
```
Returns predicted class and confidence.

## Model & Training Notes
- Use transfer learning when dataset is small.
- Apply augmentations: random crop, flip, color jitter.
- Normalize images with ImageNet mean/std if using pretrained weights.
- Monitor overfitting; use early stopping and learning-rate scheduling.

## Project Structure (suggested)
```
.
├─ data/                 # raw and processed dataset
├─ notebooks/            # experiments and visualizations
├─ src/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict.py
│  ├─ data_loader.py
│  └─ models.py
├─ django_app/          # Django web application
│  ├─ manage.py
│  ├─ requirements.txt
│  ├─ api/             # REST API endpoints
│  ├─ templates/       # HTML templates
│  ├─ static/         # CSS, JS, images
│  └─ media/          # Uploaded images
├─ models/             # saved checkpoints
├─ requirements.txt
└─ README.md
```

## Tips
- Start with a small subset to validate pipeline.
- Use mixed precision and multiple GPUs for speed.
- Log experiments (Weights & Biases, TensorBoard).

## Contributing
Submit issues and pull requests. Include unit tests for new modules and keep changes modular.

## License
MIT License — see LICENSE file.
