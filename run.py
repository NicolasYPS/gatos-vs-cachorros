# run.py
import subprocess
import sys
import os

def main():
    print("🚀 Iniciando o sistema de classificação Gato vs Cachorro...\n")

    # Caminho da imagem de teste
    image_path = "exemplos/gatos.jpeg"

    # 1. Executa o predict.py com a imagem
    print(f"1️⃣ Executando classificação da imagem: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ Erro: Arquivo não encontrado: {image_path}")
        print("Verifique se a imagem está na pasta 'exemplos/'")
        return
    try:
        subprocess.run([sys.executable, "predict.py", image_path], check=True)
        print("✅ Classificação concluída.\n")
    except subprocess.CalledProcessError:
        print("❌ Falha ao executar predict.py")
        return

    # 2. Abre a interface gráfica
    print("2️⃣ Abrindo interface gráfica...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
        print("✅ Interface fechada. Sistema finalizado.")
    except subprocess.CalledProcessError:
        print("❌ Falha ao abrir app.py")

if __name__ == "__main__":
    main()