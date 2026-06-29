import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from dotenv import load_dotenv
from src.data_loader import UTKFaceDataset, train_transforms

class FineTuneResNet(nn.Module):
    def __init__(self):
        super(FineTuneResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Congelamos todo al inicio
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Descongelamos solo los últimos bloques
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
            
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.gender_head = nn.Linear(num_ftrs, 1)
        self.age_head = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.gender_head(features), self.age_head(features)

def ejecutar_finetuning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Entrenando Experimento 5 (Fine-tuning) en: {device} ---")
    
    load_dotenv()
    ruta_dataset = os.getenv('UTKFACE_DIR')
    dataset_train_full = UTKFaceDataset(root_dir=ruta_dataset, transform=train_transforms)
    train_loader = DataLoader(dataset_train_full, batch_size=32, shuffle=True)

    model = FineTuneResNet().to(device)
    
    # Solo optimizamos los parámetros que tienen requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_age = nn.MSELoss()

    print("DEBUG: Optimizador configurado, entrando al bucle...")
    
    model.train()
    for epoch in range(10): # Hagamos 3 épocas para probar
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
    ejecutar_finetuning()

torch.save(model.state_dict(), "modelo_final.pth")
print("Modelo guardado como modelo_final.pth")
