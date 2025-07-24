# run.py
import subprocess
import sys
import os
import time

def baixar_modelo():
    """Garante que o modelo está baixado antes de prosseguir"""
    print("🔄 Verificando se o modelo está disponível...")

    modelo_destino = "models/melhor_modelo_inceptionv3.pth"

    # Verifica se a pasta models existe
    os.makedirs("models", exist_ok=True)

    # Verifica se o modelo já existe
    if os.path.exists(modelo_destino):
        print("✅ Modelo já está presente")
        return True

    print("📥 Modelo não encontrado. Iniciando download...")

    # Tenta baixar o modelo do Hugging Face
    try:
        print("🔄 Baixando modelo do Hugging Face...")
        subprocess.run([
            sys.executable, "-c", '''
import os
from huggingface_hub import hf_hub_download

print("🔹 Baixando modelo do Hugging Face...")
model_path = hf_hub_download(
    repo_id="NicolasYPS/gatos-vs-cachorros-inceptionv3",
    filename="melhor_modelo_inceptionv3.pth",
    local_dir="models",
    local_dir_use_symlinks=False
)
print(f"✅ Modelo baixado para: {model_path}")
'''
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Falha ao baixar do Hugging Face: {e}")

    # Se falhar, tenta do Google Drive
    try:
        print("🔄 Tentando baixar do Google Drive...")
        subprocess.run([
            sys.executable, "-c", '''
import os
import requests
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        with tqdm(desc=os.path.basename(destination), unit='B', unit_scale=True, ncols=80) as pbar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"✅ Modelo baixado para: {destination}")

print("🔹 Baixando modelo do Google Drive...")
os.makedirs("models", exist_ok=True)
download_file_from_google_drive("13oC9OkDUBnGGyQOWKH78oEbTDBzHgjLV", "models/melhor_modelo_inceptionv3.pth")
'''
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Falha ao baixar do Google Drive: {e}")
        print("\n💡 Dica: Baixe manualmente o modelo e salve em: models/melhor_modelo_inceptionv3.pth")
        print("   Hugging Face: https://huggingface.co/NicolasYPS/gatos-vs-cachorros-inceptionv3")
        print("   Google Drive: https://drive.google.com/file/d/13oC9OkDUBnGGyQOWKH78oEbTDBzHgjLV/view")
        return False

def main():
    print("🚀 Iniciando o sistema de classificação Gato vs Cachorro...\n")

    # Baixa o modelo se necessário
    if not baixar_modelo():
        print("❌ Não foi possível baixar o modelo. O sistema não pode continuar.")
        return

    time.sleep(1)

    # Teste automático com imagem
    image_path = "exemplos/gato.jpeg"
    if os.path.exists(image_path):
        print(f"\n1️⃣ Executando classificação da imagem: {image_path}")
        try:
            subprocess.run([sys.executable, "predict.py", image_path], check=True)
            print("✅ Classificação concluída.\n")
        except subprocess.CalledProcessError:
            print("❌ Falha ao executar predict.py")
    else:
        print(f"⚠️  Imagem de exemplo não encontrada: {image_path}")
        print("   Você ainda pode usar a interface gráfica para selecionar uma imagem.")

    # Inicia a interface gráfica
    print("2️⃣ Abrindo interface gráfica...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
        print("✅ Interface fechada. Sistema finalizado.")
    except subprocess.CalledProcessError:
        print("❌ Falha ao abrir app.py")

if __name__ == "__main__":
    main()
