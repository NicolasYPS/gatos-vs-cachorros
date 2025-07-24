# predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import sys
import os


# Adiciona o diretório atual ao caminho para importar utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa a função de download do modelo
from utils import ensure_model

# Caminho do modelo
MODEL_PATH = "models/melhor_modelo_inceptionv3.pth"

# Função para carregar o modelo
def load_model():
    print("🔄 Carregando arquitetura do modelo...")
    model = models.inception_v3(weights=None, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: gato (0), cachorro (1)
    model.eval()  # modo de avaliação
    return model

# Função de predição
def classify_image(image_path):
    # Garante que o modelo está baixado
    model_path = ensure_model()

    # Carrega o modelo
    model = load_model()
    print("🔄 Carregando pesos do modelo treinado...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("✅ Modelo carregado com sucesso!")

    # Transformações (iguais às usadas no treino)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Carrega a imagem
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Erro ao abrir imagem: {e}")
        return

    # Aplica transformações
    image_tensor = transform(image).unsqueeze(0)  # adiciona batch dimension

    # Faz a predição
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)

    # Mapeia rótulos
    classes = ["Gato", "Cachorro"]
    label = classes[predicted.item()]
    confidence_percent = confidence.item() * 100

    # Exibe resultado
    print(f"\n📸 Imagem: {image_path}")
    print(f"🎯 Predição: {label}")
    print(f"📊 Confiança: {confidence_percent:.2f}%")

if __name__ == "__main__":
    # Verifica se o caminho da imagem foi fornecido
    if len(sys.argv) != 2:
        print("❌ Use: python predict.py <caminho/para/imagem.jpg>")
        print("Exemplo: python predict.py exemplos/gato.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # Verifica se a imagem existe
    if not os.path.exists(image_path):
        print(f"❌ Arquivo não encontrado: {image_path}")
        sys.exit(1)

    # Executa a classificação
    classify_image(image_path)