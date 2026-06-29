import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv
from src.data_loader import UTKFaceDataset, train_transforms

class ResNetMultitask(nn.Module):
    def __init__(self):
        super(ResNetMultitask, self).__init__()
        # Cargar ResNet18 preentrenada
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Congelar todas las capas (Transfer Learning puro)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Modificar la última capa para nuestras dos tareas
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
        self.gender_head = nn.Linear(num_ftrs, 1)
        self.age_head = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.gender_head(features), self.age_head(features)

def ejecutar_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Entrenando Experimento 4 (ResNet Congelada) en: {device} ---")
    
    load_dotenv()
    ruta_dataset = os.getenv('UTKFACE_DIR')
    dataset_train_full = UTKFaceDataset(root_dir=ruta_dataset, transform=train_transforms)
    train_loader = DataLoader(dataset_train_full, batch_size=32, shuffle=True)
    
    model = ResNetMultitask().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_age = nn.MSELoss()

    print("DEBUG: Optimizador configurado, entrando al bucle...")
    
    model.train()
    for epoch in range(10): # Entrenemos 10 épocas
        running_loss = 0.0
        for images, ages, genders in train_loader:
            images, ages, genders = images.to(device), ages.to(device), genders.to(device)
            optimizer.zero_grad()
            
            out_gender, out_age = model(images)
            loss = criterion_gender(out_gender, genders) + 0.1 * criterion_age(out_age, ages)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Época {epoch+1} completada. Loss: {running_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    ejecutar_resnet()
