# utils.py
from huggingface_hub import hf_hub_download
import os

MODEL_PATH = "models/melhor_modelo_inceptionv3.pth"

def ensure_model():
    """Baixa o modelo do Hugging Face se não existir localmente."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Modelo encontrado: {MODEL_PATH}")
        return MODEL_PATH

    print("📥 Modelo não encontrado. Baixando do Hugging Face...")
    os.makedirs("models", exist_ok=True)

    try:
        # Baixa o modelo publicamente (sem precisar de token)
        downloaded_path = hf_hub_download(
            repo_id="NicolasYPS/gatos-vs-cachorros-inceptionv3",
            filename="melhor_modelo_inceptionv3.pth",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"✅ Modelo baixado: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        print(f"❌ Erro ao baixar o modelo: {e}")
        print("\n💡 Dica: Verifique sua conexão ou acesse:")
        print("   https://huggingface.co/NicolasYPS/gatos-vs-cachorros-inceptionv3")
        raise