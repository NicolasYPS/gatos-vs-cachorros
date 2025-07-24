# predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Caminho do modelo
MODEL_PATH = "models/melhor_modelo_inceptionv3.pth"

# Transformações
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carrega o modelo
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    model = models.inception_v3(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# Função de predição
def classify_image(image_path):
    if not os.path.exists(image_path):
        return {"error": "Imagem não encontrada"}

    try:
        model = load_model()
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            prob = torch.nn.functional.softmax(output, dim=1)[0]
            confidence = prob[predicted.item()].item() * 100

        label = "GATO" if predicted.item() == 1 else "CACHORRO"
        return {
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}