import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import cv2

# Configurações
pasta_gato = "data/gatos"
pasta_cachorro = "data/cachorro"
os.makedirs(pasta_gato, exist_ok=True)
os.makedirs(pasta_cachorro, exist_ok=True)

LIMITE_IMAGENS = 500  # Aumentado para melhor qualidade

def contar_imagens(pasta):
    return len([f for f in os.listdir(pasta) if f.startswith("img_") and f.endswith(".jpg")])

cont_gato_atual = contar_imagens(pasta_gato)
cont_cachorro_atual = contar_imagens(pasta_cachorro)

print(f"🖼️  Imagens existentes: {cont_gato_atual} gatos, {cont_cachorro_atual} cachorros")

# Função para salvar mais imagens até o limite
def completar_dataset(dataset, pasta_gato, pasta_cachorro, limite=500):
    cont_gato = contar_imagens(pasta_gato)
    cont_cachorro = contar_imagens(pasta_cachorro)
    max_gato = limite - cont_gato
    max_cachorro = limite - cont_cachorro

    print(f"🔄 Completando dataset: faltam {max_gato} gatos e {max_cachorro} cachorros...")

    for imagem, rotulo in dataset:
        if cont_gato >= limite and cont_cachorro >= limite:
            break

        # Redimensionar e converter
        img = tf.image.resize(imagem, (299, 299))
        img = tf.cast(img, tf.uint8).numpy()

        if rotulo == 0 and cont_gato < limite:
            cv2.imwrite(f"{pasta_gato}/img_{cont_gato:03d}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cont_gato += 1
        elif rotulo == 1 and cont_cachorro < limite:
            cv2.imwrite(f"{pasta_cachorro}/img_{cont_cachorro:03d}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cont_cachorro += 1

    print(f"✅ Total final: {cont_gato} gatos e {cont_cachorro} cachorros.")

# Carregar dataset completo
(ds_train), info = tfds.load('cats_vs_dogs', split='train', as_supervised=True, with_info=True)

# Executar salvamento
completar_dataset(ds_train, pasta_gato, pasta_cachorro, limite=LIMITE_IMAGENS)