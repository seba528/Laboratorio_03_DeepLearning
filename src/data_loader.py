import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [f for f in os.listdir(root_dir) if f.lower().endswith('.jpg')]
        print(f"DEBUG: Imágenes válidas cargadas: {len(self.image_paths)}")
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        parts = img_name.split('_')
        age = float(parts[0])      
        gender = float(parts[1])   
        return image, torch.tensor([age], dtype=torch.float32), torch.tensor([gender], dtype=torch.float32)

# ESTOS SON LOS NOMBRES QUE TUS SCRIPTS BUSCAN
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
