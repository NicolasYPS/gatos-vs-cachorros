# predict.py
from PIL import Image
import numpy as np
import sys
import os

# Garante que o caminho do módulo está certo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ensure_model

def classify_image(image_path):
    # ✅ Importação tardia do TensorFlow/Keras (só carrega quando necessário)
    from tensorflow.keras.models import load_model

    # Garante que o modelo está baixado
    model_path = ensure_model()
    print("🔄 Carregando modelo...")
    model = load_model(model_path)

    # Carrega e processa a imagem
    try:
        img = Image.open(image_path).resize((224, 224))
    except Exception as e:
        print(f"❌ Erro ao abrir imagem: {e}")
        return

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predição
    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "Cachorro" if prediction > 0.5 else "Gato"
    confidence = (prediction if prediction > 0.5 else 1 - prediction) * 100

    print(f"\n📸 Imagem: {image_path}")
    print(f"🎯 Resultado: {label}")
    print(f"📊 Confiança: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❌ Use: python predict.py <caminho/para/imagem.jpg>")
        print("Exemplo: python predict.py exemplos/gato.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"❌ Arquivo não encontrado: {image_path}")
        sys.exit(1)

    classify_image(image_path)