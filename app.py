# app.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  
from predict import classify_image

# Variáveis globais para a imagem
imagem_label = None
imagem_tk = None

# Função chamada ao clicar em "Carregar Imagem"
def selecionar_imagem():
    global imagem_label, imagem_tk

    caminho = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
    )
    if caminho:
        resultado = classify_image(caminho)
        if "error" in resultado:
            messagebox.showerror("Erro", resultado["error"])
        else:
            resultado_var.set(f"{resultado['label']}\n({resultado['confidence']:.1f}%)")

            # Exibe a imagem
            imagem = Image.open(caminho)
            imagem = imagem.resize((200, 200))  # Redimensiona para caber
            imagem_tk = ImageTk.PhotoImage(imagem)

            if imagem_label is None:
                imagem_label = tk.Label(root, image=imagem_tk, bg="white")
                imagem_label.pack(pady=10)
            else:
                imagem_label.configure(image=imagem_tk)
                imagem_label.image = imagem_tk  # Mantém referência

# Cria a janela
root = tk.Tk()
root.title("🐱🐶 Gato ou Cachorro")
root.geometry("400x500")  # aumentei altura para caber imagem
root.resizable(False, False)
root.configure(bg="white")

# Título
tk.Label(root, text="Classificador de Imagens", font=("Arial", 16, "bold"), bg="white").pack(pady=20)

# Resultado
resultado_var = tk.StringVar(value="Escolha uma imagem")
tk.Label(root, textvariable=resultado_var, font=("Arial", 16), bg="white", fg="black").pack(pady=10)

# Botões
frame_botoes = tk.Frame(root, bg="white")
frame_botoes.pack(pady=20)

tk.Button(
    frame_botoes,
    text="📁 Carregar Imagem",
    font=("Arial", 12),
    bg="#4CAF50",
    fg="white",
    width=15,
    command=selecionar_imagem
).pack(side="left", padx=10)

tk.Button(
    frame_botoes,
    text="🚪 Sair",
    font=("Arial", 12),
    bg="#f44336",
    fg="white",
    width=10,
    command=root.destroy
).pack(side="left", padx=10)

# Inicia a GUI
root.mainloop()
