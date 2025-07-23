# Gato ou Cachorro - Classificador de Imagens com IA

Um modelo de aprendizado de máquina treinado para classificar imagens como **gato** ou **cachorro** usando transferência de aprendizado com o modelo **InceptionV3**.

Conhecimento obtido via Bootcamp bairesdev machine learning training e DIO.

---

## 📌 Visão Geral

Este projeto é um classificador binário de imagens que distingue entre fotos de **gatos** e **cachorros**. Foi desenvolvido como um exercício clássico de visão computacional e deep learning, utilizando transferência de aprendizado para alcançar alta acurácia com um conjunto de dados relativamente pequeno.

- **Modelo base:** InceptionV3 (pré-treinado no ImageNet)
- **Dataset:** 1000 imagens (500 gatos, 500 cachorros)
- **Framework:** TensorFlow/Keras
- **Métricas de validação:**  
  - **Perda (Loss):** 0.0108  
  - **Acurácia:** 99.50%

---

## 🚀 Funcionalidades

- Classificação em tempo real de imagens de gatos e cachorros
- Transferência de aprendizado com congelamento de camadas e fine-tuning
- Pré-processamento de imagens (redimensionamento, normalização)
- Validação robusta com conjunto de teste separado

---

## 📦 Requisitos

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib (para visualização)
- Pillow (PIL)

Instale as dependências com:

```bash
pip install -r requirements.txt
```


## Autor

Nicolas Souza  
[LinkedIn](https://linkedin.com/in/nicolas-y-p-souza) | [GitHub](https://github.com/NicolasYPS)
