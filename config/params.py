import torch

# Path configurations
DATA_PATH = "./data/"
LOG_DIR = "./logs/"
MODEL_SAVE_PATH = "./saved_models/best_model.pth"

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base model configurations
BASE_MODELS = {
    "resnet": "resnet50",
    "densenet": "densenet169",
    "efficientnet": "efficientnet_b0"
}

# Training parameters
TRAIN_PARAMS = {
    "batch_size": 32,
    "num_epochs": 200,
    "initial_lr": 3e-4,
    "weight_decay": 1e-5,
    "sparsity_reg": 0.01,
    "patience": 15,
    "gamma": 2.0,
    "alpha": [0.9, 0.1]
}

# Augmentation parameters
AUGMENTATION_PARAMS = {
    "resize_dim": 256,
    "crop_size": 224,
    "blur_sigma": (0, 3.0),
    "noise_scale": 0.05,
    "brightness_range": (-30, 30),
    "contrast_range": (0.5, 2.0)
}

# Model architecture parameters
ENSEMBLE_PARAMS = {
    "meta_input_dim": 3,
    "meta_hidden_dims": [512, 256, 128, 64],
    "dropout_rate": 0.1,
    "output_activation": "sigmoid"
}
