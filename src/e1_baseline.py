import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from dotenv import load_dotenv

# --- LA CORRECCIÓN ESTÁ AQUÍ ---
# Ahora le decimos explícitamente a Python que busque dentro de la carpeta 'src'
from src.data_loader import UTKFaceDataset, train_transforms, test_transforms

def extract_features_and_labels(dataloader, desc="Procesando"):
    """
    Itera sobre un DataLoader de PyTorch y convierte los tensores de imágenes
    en matrices planas de Numpy para scikit-learn.
    """
    X_list, age_list, gender_list = [], [], []
    
    print(f"--- Extrayendo datos: {desc} ---")
    start_time = time.time()
    
    for batch_idx, (images, ages, genders) in enumerate(dataloader):
        # Aplanar la imagen RGB (Batch, 3, 224, 224) a (Batch, 150528)
        X_flat = images.view(images.size(0), -1).numpy()
        
        X_list.append(X_flat)
        age_list.append(ages.numpy())
        gender_list.append(genders.numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Lotes procesados: {batch_idx + 1}/{len(dataloader)}")
            
    # Concatenar todo en grandes matrices de Numpy
    X = np.vstack(X_list)
    y_age = np.vstack(age_list).ravel()
    y_gender = np.vstack(gender_list).ravel()
    
    print(f"Extracción completada en {time.time() - start_time:.2f} segundos.")
    print(f"Dimensión de X: {X.shape}")
    
    return X, y_age, y_gender


def ejecutar_baseline():
    # 1. CARGAR RUTA DESDE EL ARCHIVO .ENV
    load_dotenv()
    ruta_dataset = os.getenv('UTKFACE_DIR')
    
    if not ruta_dataset or not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"⚠️ ¡Error! No se encontró la carpeta en la ruta: {ruta_dataset}. Revisa tu archivo .env en la raíz del proyecto.")

    print(f"📂 Cargando dataset desde: {ruta_dataset}")
    
    # 2. PREPARACIÓN DE DATOS
    dataset_train_full = UTKFaceDataset(root_dir=ruta_dataset, transform=train_transforms)
    dataset_test_full = UTKFaceDataset(root_dir=ruta_dataset, transform=test_transforms)

    # Dividir 80% Train / 20% Test (Semilla fija para reproducibilidad)
    total_size = len(dataset_train_full)
    torch.manual_seed(42)
    indices = torch.randperm(total_size).tolist()
    train_size = int(0.8 * total_size)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_dataset = Subset(dataset_train_full, train_idx)
    test_dataset = Subset(dataset_test_full, test_idx)

    # Batch Size grande para acelerar la extracción en el servidor
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Extraer las matrices a la memoria RAM del servidor
    X_train, y_age_train, y_gender_train = extract_features_and_labels(train_loader, "Train")
    X_test, y_age_test, y_gender_test = extract_features_and_labels(test_loader, "Test")

    # 3. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
    print("\n--- Iniciando PCA ---")
    n_components = 100 
    pca = PCA(n_components=n_components, random_state=42)
    
    start_time = time.time()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA completado en {time.time() - start_time:.2f} segundos.")
    print(f"Varianza explicada por {n_components} componentes: {np.sum(pca.explained_variance_ratio_):.2%}")

    # 4. ENTRENAMIENTO: CLASIFICACIÓN DE GÉNERO
    print("\n--- Entrenando Clasificador de Género (Regresión Logística) ---")
    clf_gender = LogisticRegression(max_iter=1000)
    clf_gender.fit(X_train_pca, y_gender_train)
    
    y_pred_gender = clf_gender.predict(X_test_pca)
    acc_gender = accuracy_score(y_gender_test, y_pred_gender)
    f1_gender = f1_score(y_gender_test, y_pred_gender)
    print(f"✅ Género - Accuracy: {acc_gender:.4f} | F1-Score: {f1_gender:.4f}")

    # 5. ENTRENAMIENTO: REGRESIÓN DE EDAD
    print("\n--- Entrenando Regresor de Edad (Ridge) ---")
    reg_age = Ridge(alpha=1.0)
    reg_age.fit(X_train_pca, y_age_train)
    
    y_pred_age = reg_age.predict(X_test_pca)
    mse_age = mean_squared_error(y_age_test, y_pred_age)
    mae_age = mean_absolute_error(y_age_test, y_pred_age)
    print(f"✅ Edad - MSE: {mse_age:.4f} | MAE: {mae_age:.4f} años")


if __name__ == "__main__":
    ejecutar_baseline()
