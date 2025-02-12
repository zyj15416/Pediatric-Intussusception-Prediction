import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from config import params as cfg
from data.data_loader import get_dataloaders
from models.base_models import BaseModel
from models.ensemble import RegularizedEnsemble, MetaLearner
from models.losses import composite_loss
from utils.logger import TrainingLogger, GradientAccumulator
from utils.metrics import calculate_metrics

def train_ensemble():
    # Initialize components
    train_loader, val_loader = get_dataloaders()
    logger = TrainingLogger(cfg.LOG_DIR)
    
    # Create base models
    base_models = [BaseModel(arch).to(cfg.DEVICE) for arch in cfg.BASE_MODELS.values()]
    
    # Create meta model
    meta_model = MetaLearner(cfg.ENSEMBLE_PARAMS['meta_input_dim']).to(cfg.DEVICE)
    
    # Create ensemble
    model = RegularizedEnsemble(base_models, meta_model).to(cfg.DEVICE)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                          lr=cfg.TRAIN_PARAMS['initial_lr'],
                          weight_decay=cfg.TRAIN_PARAMS['weight_decay'])
    
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.TRAIN_PARAMS['num_epochs'],
        eta_min=1e-6
    )
    
    # Training loop
    best_auc = 0.0
    for epoch in range(cfg.TRAIN_PARAMS['num_epochs']):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(cfg.DEVICE)
            labels = labels.float().to(cfg.DEVICE)
            
            outputs = model(inputs)
            loss = composite_loss(outputs.squeeze(), labels, model)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs, val_labels = [], []
            for inputs, labels in val_loader:
                inputs = inputs.to(cfg.DEVICE)
                outputs = model(inputs)
                val_outputs.append(outputs.cpu())
                val_labels.append(labels.cpu())
            
            val_metrics = calculate_metrics(
                torch.cat(val_outputs).numpy(),
                torch.cat(val_labels).numpy()
            )
        
        # Log metrics
        logger.log_metrics('val', val_metrics, epoch)
        logger.log_learning_rate(scheduler.get_last_lr()[0], epoch)
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
        
        scheduler.step()
    
    logger.close()

if __name__ == '__main__':
    train_ensemble()
