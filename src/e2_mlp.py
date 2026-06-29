import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

# --- LA CORRECCIÓN DE IMPORTACIÓN ---
from src.data_loader import UTKFaceDataset, train_transforms, test_transforms

# 1. DEFINIR LA ARQUITECTURA MLP MULTITAREA
class MultitaskMLP(nn.Module):
    def __init__(self, input_size=224*224*3):
        super(MultitaskMLP, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Representación Compartida (Shared Backbone)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Cabezas Multitarea
        self.gender_head = nn.Linear(128, 1) # Clasificación Binaria
        self.age_head = nn.Linear(128, 1)    # Regresión

    def forward(self, x):
        x = self.flatten(x)                
        shared_features = self.shared_layers(x) 
        
        out_gender = self.gender_head(shared_features)
        out_age = self.age_head(shared_features)
        
        return out_gender, out_age

# 2. FUNCIÓN DE ENTRENAMIENTO
def ejecutar_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Entrenando Experimento 2 (MLP) usando: {device} ---")

    # --- CORRECCIÓN DE LA RUTA DEL DATASET (Usando .env) ---
    load_dotenv()
    ruta_dataset = os.getenv('UTKFACE_DIR')
    
    if not ruta_dataset or not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"⚠️ Error: No se encontró la ruta {ruta_dataset}. Revisa tu archivo .env")

    print(f"📂 Cargando dataset desde: {ruta_dataset}")

    # Preparar el Dataset
    dataset_train_full = UTKFaceDataset(root_dir=ruta_dataset, transform=train_transforms)
    total_size = len(dataset_train_full)
    torch.manual_seed(42)
    indices = torch.randperm(total_size).tolist()
    train_idx = indices[:int(0.8 * total_size)]
    train_dataset = Subset(dataset_train_full, train_idx)
    
    # Batch size pequeño porque el MLP consume muchísima RAM/VRAM
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Inicializar modelo 
    model = MultitaskMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Funciones de Pérdida
    criterion_gender = nn.BCEWithLogitsLoss() 
    criterion_age = nn.MSELoss()              
    lambda_age = 0.1 

    # Bucle de entrenamiento
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for images, ages, genders in train_loader:
            images, ages, genders = images.to(device), ages.to(device), genders.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out_gender, out_age = model(images)
            
            loss_gender = criterion_gender(out_gender, genders)
            loss_age = criterion_age(out_age, ages)
            
            total_loss = loss_gender + (lambda_age * loss_age)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        tiempo_epoca = time.time() - start_time
        print(f"Época [{epoch+1}/{epochs}] | Loss Total: {running_loss/len(train_loader):.4f} | Tiempo: {tiempo_epoca:.2f}s")

if __name__ == "__main__":
    ejecutar_mlp()
