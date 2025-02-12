import torchvision.models as models
import torch.nn as nn
from config import params as cfg

class BaseModel(nn.Module):
    def __init__(self, arch, pretrained=True):
        super().__init__()
        self.base_model = self._create_base_model(arch, pretrained)
        self._freeze_layers()
        
    def _create_base_model(self, arch, pretrained):
        if arch == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, 1)
        elif arch == 'densenet169':
            model = models.densenet169(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, 1)
        elif arch == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        return model
    
    def _freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        if isinstance(self.base_model, models.ResNet):
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True
        elif isinstance(self.base_model, models.DenseNet):
            for param in self.base_model.features[-1].parameters():
                param.requires_grad = True
        elif isinstance(self.base_model, models.EfficientNet):
            for param in self.base_model.features[-1].parameters():
                param.requires_grad = True
                
    def forward(self, x):
        return torch.sigmoid(self.base_model(x))
