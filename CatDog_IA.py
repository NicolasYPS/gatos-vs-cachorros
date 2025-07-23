import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Carregar modelo treinado
def carregar_modelo():
    try:
        model = models.inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: gato e cachorro
        model.load_state_dict(torch.load("melhor_modelo_inceptionv3.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        messagebox.showerror("Erro", f"❌ Falha ao carregar o modelo: {e}")
        return None

# Transformações
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Função para classificação
def classificar_imagem(caminho, model, label_resultado, panel):
    try:
        # Carregar e transformar imagem
        image = Image.open(caminho).convert("RGB")
        image.thumbnail((300, 300))  # Reduzir para exibir na GUI
        img = transform(image).unsqueeze(0).to("cpu")

        # Previsão
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            classe = predicted.item()
            confianca = torch.nn.functional.softmax(output, dim=1)[0][classe].item() * 100

        # Atualizar GUI
        resultado = f"{'🐶 CACHORRO' if classe == 0 else '🐱 GATO'} (confiança: {confianca:.2f}%)"
        label_resultado.config(text=resultado)

        # Redimensionar para exibir
        display_img = Image.open(caminho).convert("RGB")
        display_img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(display_img)
        panel.configure(image=img_tk)
        panel.image = img_tk  # evitar coleta de lixo
    except Exception as e:
        label_resultado.config(text=f"❌ Erro: {e}")

# Função para selecionar imagem
def selecionar_imagem(model, label_resultado, panel):
    caminho = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if caminho:
        classificar_imagem(caminho, model, label_resultado, panel)

# GUI principal
def main():
    # Carregar modelo
    model = carregar_modelo()
    if not model:
        return

    # Criar janela
    root = tk.Tk()
    root.title("🐾 Gato vs Cachorro")
    root.geometry("600x400")
    root.resizable(False, False)
    root.configure(bg="white")

    # Label para resultado
    label_resultado = tk.Label(root, text="Escolha uma imagem", font=("Arial", 14), bg="white")
    label_resultado.pack(pady=10)

    # Painel para imagem
    panel = tk.Label(root, bg="white")
    panel.pack()

    # Botão para carregar imagem
    btn = tk.Button(
        root,
        text="📁 Carregar Imagem",
        font=("Arial", 12),
        command=lambda: selecionar_imagem(model, label_resultado, panel)
    )
    btn.pack(pady=10)

    # Rodar GUI
    root.mainloop()

if __name__ == "__main__":
    main()