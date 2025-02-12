import os
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import imgaug.augmenters as iaa

class MedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        self.samples = []
        
        for label in ['success', 'failed']:
            label_dir = os.path.join(root_dir, label)
            for xml_file in os.listdir(label_dir):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(label_dir, xml_file)
                    img_path = xml_path.replace('.xml', '.jpg')
                    if not os.path.exists(img_path):
                        img_path = img_path.replace('.jpg', '.png')
                    if os.path.exists(img_path):
                        self.samples.append((img_path, xml_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path, label = self.samples[idx]
        
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        
        # Parse XML annotations
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        coords = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
        img = img.crop(coords)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        # Convert label
        label = 0 if label == 'success' else 1
        
        return img, label

def get_dataloaders():
    from config import params as cfg
    
    train_transform = transforms.Compose([
        transforms.Resize(cfg.AUGMENTATION_PARAMS['resize_dim']),
        transforms.RandomCrop(cfg.AUGMENTATION_PARAMS['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(cfg.AUGMENTATION_PARAMS['resize_dim']),
        transforms.CenterCrop(cfg.AUGMENTATION_PARAMS['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MedicalDataset(os.path.join(cfg.DATA_PATH, 'train'), train_transform)
    val_dataset = MedicalDataset(os.path.join(cfg.DATA_PATH, 'val'), val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_PARAMS['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN_PARAMS['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
