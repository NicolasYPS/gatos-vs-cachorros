# utils.py
import os
import requests
from tqdm import tqdm

# ID do seu arquivo no Google Drive
MODEL_ID = "13oC9OkDUBnGGyQOWKH78oEbTDBzHgjLV"
MODEL_PATH = "models/melhor_modelo_inceptionv3.pth"  # Nome correto do modelo PyTorch

def download_file_from_google_drive(id, destination):
    """Baixa arquivo do Google Drive tratando o aviso de 'conteúdo sensível'"""
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    """Extrai o token de confirmação do cookie do Google Drive"""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    """Salva o conteúdo com barra de progresso"""
    chunk_size = 32768
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit="B",
        unit_scale=True,
        ncols=80
    ) as pbar:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def ensure_model():
    """Garante que o modelo do PyTorch está baixado. Se não, baixa do Google Drive."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Modelo encontrado: {MODEL_PATH}")
        return MODEL_PATH

    print("📥 Modelo não encontrado. Baixando do Google Drive...")
    os.makedirs("models", exist_ok=True)

    try:
        download_file_from_google_drive(MODEL_ID, MODEL_PATH)
        print(f"✅ Modelo baixado: {MODEL_PATH}")
        return MODEL_PATH
    except Exception as e:
        print(f"❌ Erro ao baixar o modelo: {e}")
        print("\n💡 Dica: Baixe manualmente em:")
        print(f"   https://drive.google.com/uc?export=download&id={MODEL_ID}")
        print(f"   e salve em: {MODEL_PATH}")
        raise