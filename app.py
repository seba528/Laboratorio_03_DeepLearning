import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.e5_finetune import FineTuneResNet # Importamos la clase de tu modelo

# 1. Configuración del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FineTuneResNet().to(device)
model.load_state_dict(torch.load("modelo_final.pth", map_location=device))
model.eval()

# 2. Transformaciones (deben ser iguales a test_transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Interfaz Streamlit
st.title("IA: Predicción de Edad y Género")
uploaded_file = st.file_uploader("Sube una foto de un rostro...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    # Procesar imagen
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inferencia
    with torch.no_grad():
        gender_out, age_out = model(input_tensor)
        gender_pred = "Mujer" if torch.sigmoid(gender_out) > 0.5 else "Hombre"
        age_pred = age_out.item()
    
    st.subheader(f"Género: {gender_pred}")
    st.subheader(f"Edad estimada: {int(age_pred)} años")
