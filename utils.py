# utils.py
import os
import requests
from tqdm import tqdm

MODEL_URL = "https://drive.google.com/uc?export=download&id=13oC9OkDUBnGGyQOWKH78oEbTDBzHgjLV"
MODEL_PATH = "models/modelo.h5"

def ensure_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Modelo encontrado: {MODEL_PATH}")
        return MODEL_PATH

    print("📥 Modelo não encontrado. Baixando automaticamente...")
    os.makedirs("models", exist_ok=True)

    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(MODEL_PATH, 'wb') as f, tqdm(
            desc="modelo.h5",
            total=total_size,
            unit='B',
            unit_scale=True,
            ncols=80
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"✅ Modelo baixado: {MODEL_PATH}")
        return MODEL_PATH

    except Exception as e:
        print(f"❌ Erro ao baixar o modelo: {e}")
        print("\n💡 Dica: Baixe manualmente em:")
        print("   https://drive.google.com/file/d/13oC9OkDUBnGGyQOWKH78oEbTDBzHgjLV/view")
        print(f"   e salve em: {MODEL_PATH}")
        raise