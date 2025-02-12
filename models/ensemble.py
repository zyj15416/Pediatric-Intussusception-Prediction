import torch.nn as nn
from config import params as cfg

class MetaLearner(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SELU(),
            nn.AlphaDropout(cfg.ENSEMBLE_PARAMS['dropout_rate']),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.AlphaDropout(cfg.ENSEMBLE_PARAMS['dropout_rate']),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                
    def forward(self, x):
        return self.layers(x)

class RegularizedEnsemble(nn.Module):
    def __init__(self, base_models, meta_model):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        
    def forward(self, x):
        base_outputs = [model(x) for model in self.base_models]
        concatenated = torch.cat(base_outputs, dim=1)
        final_output = self.meta_model(concatenated)
        return final_output
    
    def regularization_loss(self):
        l1_reg = sum(param.abs().sum() for param in self.meta_model.parameters())
        sparsity_reg = sum(torch.norm(param, p=2) for param in self.meta_model.parameters())
        return cfg.TRAIN_PARAMS['weight_decay'] * l1_reg + cfg.TRAIN_PARAMS['sparsity_reg'] * sparsity_reg
