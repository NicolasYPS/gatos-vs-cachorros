import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import cv2

# Definir pastas
pasta_gato = "gatos"
pasta_cachorro = "cachorro"
os.makedirs(pasta_gato, exist_ok=True)
os.makedirs(pasta_cachorro, exist_ok=True)

# Carregar dataset
(ds_train, ds_test), info = tfds.load(
    'cats_vs_dogs', 
    split=['train[:80%]', 'train[80%:]'], 
    with_info=True, 
    as_supervised=True
)

# Função para salvar imagens em pastas
def salvar_imagens(dataset, pasta_gato, pasta_cachorro):
    cont_gato = 0
    cont_cachorro = 0
    for imagem, rotulo in dataset:
        img = tf.keras.preprocessing.image.array_to_img(imagem.numpy())
        img = img.resize((150, 150))  # Redimensionar (opcional)
        img_array = np.array(img) / 255.0  # Normalizar (opcional)

        if rotulo == 0:  # Gato
            cv2.imwrite(f"{pasta_gato}/img_{cont_gato:03d}.jpg", 
                        cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cont_gato += 1
        else:  # Cachorro
            cv2.imwrite(f"{pasta_cachorro}/img_{cont_cachorro:03d}.jpg", 
                        cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cont_cachorro += 1

        if cont_gato >= 100 and cont_cachorro >= 100:
            break

    print(f"✅ {cont_gato} imagens de gatos salvas em {pasta_gato}")
    print(f"✅ {cont_cachorro} imagens de cachorros salvas em {pasta_cachorro}")

# Executar salvamento
salvar_imagens(ds_train, pasta_gato, pasta_cachorro)