# Pediatric-Intussusception-Prediction
.data/data_loader.py - Implements custom dataset handling with medical image loading, XML annotation parsing, and data augmentation pipelines.

.models/base_models.py - Defines base architectures (ResNet/DenseNet/EfficientNet) with customized final layers and partial layer freezing.

.models/ensemble.py - Implements the stacked ensemble architecture with meta-learner and regularization mechanisms for model fusion.

.models/losses.py - Contains custom loss functions combining focal loss with regularization terms for handling class imbalance.

.train.py - Orchestrates the complete training workflow including model initialization, optimization, and checkpoint saving.

.inference.py - Provides production-ready prediction capabilities with pre-processing.
