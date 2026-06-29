import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

# Importaciones correctas desde src
from src.data_loader import UTKFaceDataset, train_transforms, test_transforms

# 1. DEFINIR LA ARQUITECTURA CNN MULTITAREA
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # --- Shared Backbone (Convolucional) ---
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduce a 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduce a 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Reduce a 28x28
        )
        
        # Capa para aplanar tras las convoluciones
        self.flatten = nn.Flatten()
        
        # OJO: La entrada a esta capa densa depende del tamaño de tu imagen.
        # Si entran imágenes de 224x224 y pasamos por 3 MaxPools (dividido por 2 cada uno = /8),
        # el mapa de características final es de 28x28.
        self.fc = nn.Linear(128 * 28 * 28, 512)
        
        # --- Cabezas Multitarea ---
        self.gender_head = nn.Linear(512, 1) # Clasificación Binaria
        self.age_head = nn.Linear(512, 1)    # Regresión

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        
        gender_out = self.gender_head(x)
        age_out = self.age_head(x)
        
        return gender_out, age_out

# 2. FUNCIÓN DE ENTRENAMIENTO (¡La que faltaba!)
def ejecutar_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Entrenando Experimento 3 (CNN Simple) usando: {device} ---")

    # Cargar rutas
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
    
    # Podemos usar un batch size ligeramente mayor en CNNs comparado a MLPs gigantes
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Inicializar modelo
    model = SimpleCNN().to(device)
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
    ejecutar_cnn()
