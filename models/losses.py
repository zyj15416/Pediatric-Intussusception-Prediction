import torch.nn as nn
import torch.nn.functional as F
from config import params as cfg

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=cfg.TRAIN_PARAMS['alpha'], gamma=cfg.TRAIN_PARAMS['gamma']):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        
        alpha_t = self.alpha[0] * (1 - targets) + self.alpha[1] * targets
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        return focal_loss.mean()

def composite_loss(output, target, model):
    focal_loss = AdaptiveFocalLoss()(output, target)
    reg_loss = model.regularization_loss()
    return focal_loss + reg_loss
