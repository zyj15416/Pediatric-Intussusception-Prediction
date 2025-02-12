import torch
from torchvision import transforms
from config import params as cfg
from models.base_models import BaseModel
from models.ensemble import RegularizedEnsemble, MetaLearner

class EnsemblePredictor:
    def __init__(self, model_path=cfg.MODEL_SAVE_PATH):
        self.device = torch.device(cfg.DEVICE)
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(cfg.AUGMENTATION_PARAMS['resize_dim']),
            transforms.CenterCrop(cfg.AUGMENTATION_PARAMS['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path):
        # Create base models
        base_models = [BaseModel(arch).to(self.device) for arch in cfg.BASE_MODELS.values()]
        
        # Create meta model
        meta_model = MetaLearner(cfg.ENSEMBLE_PARAMS['meta_input_dim']).to(self.device)
        
        # Create ensemble
        model = RegularizedEnsemble(base_models, meta_model).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def predict(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(image_tensor)
        return prediction.item()

class HeatmapGenerator:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        
    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for base models
        for i, model in enumerate(self.model.base_models):
            model.base_model.layer4[-1].register_forward_hook(get_activation(f'base_{i}'))
            
    def generate_heatmap(self, image):
        self.activations.clear()
        self.model(image.unsqueeze(0))
        # Generate heatmap using activation maps
        # Implementation depends on specific visualization requirements
        return combined_heatmap
