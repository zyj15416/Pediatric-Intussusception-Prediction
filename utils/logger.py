from torch.utils.tensorboard import SummaryWriter
import time

class TrainingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()
        
    def log_metrics(self, phase, metrics, epoch):
        for metric, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{metric}', value, epoch)
            
    def log_learning_rate(self, lr, epoch):
        self.writer.add_scalar('LR', lr, epoch)
        
    def log_weights(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, epoch)
            
    def close(self):
        self.writer.close()
        print(f"Training completed in {time.time() - self.start_time:.2f}s")

class GradientAccumulator:
    def __init__(self, accumulation_steps):
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        
    def should_update(self):
        return (self.step_counter % self.accumulation_steps) == 0
    
    def step(self):
        self.step_counter += 1
