import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import Inception_V3_Weights
import matplotlib.pyplot as plt
from time import time
import os

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Transformações com aumento de dados
transform_train = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformações para validação (sem aumento de dados)
transform_val = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregar dataset
dataset = datasets.ImageFolder(root='data', transform=None)

# Dividir em treino e validação
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Aplicar transformações específicas
train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val

# Calcular batch_size para 10 passos por época
steps_per_epoch = 10
batch_size = len(dataset) // steps_per_epoch  # 1000 // 10 = 100

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Modelo InceptionV3
weights = Inception_V3_Weights.IMAGENET1K_V1
model = models.inception_v3(weights=weights, aux_logits=True)  # Carrega com aux_logits
model.aux_logits = False
model.AuxLogits = None
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: gato (0) e cachorro (1)
model = model.to(device)

# Otimizador e critério
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Listas para métricas
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Early stopping
best_val_loss = float('inf')
patience = 15
trigger_times = 0
best_model_path = "melhor_modelo_inceptionv3.pth"

# Treinamento
EPOCHS = 100
inicio = time()

print(f"\nTotal de imagens: {len(dataset)}")
print(f"Batch size: {batch_size} (passos por época: {steps_per_epoch})")
print("Iniciando treinamento com InceptionV3...\n")

for epoch in range(EPOCHS):
    # --- Treinamento ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"\nÉpoca [{epoch+1}/{EPOCHS}]")

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # ✅ Print a cada passo
        print(f"  → Passo [{step+1}/{steps_per_epoch}] - Perda: {loss.item():.4f} | Acurácia: {100 * (predicted == labels).sum().item() / labels.size(0):.2f}%")

    # Média da época
    epoch_train_loss = running_loss / steps_per_epoch
    epoch_train_acc = 100 * correct_train / (steps_per_epoch * batch_size)

    # --- Validação ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * correct_val / total_val

    # Armazenar métricas
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_acc)
    val_accuracies.append(epoch_val_acc)

    # Mostra status no terminal
    print(f"\n  ➤ Média da época - Treino: Perda {epoch_train_loss:.4f} | Acurácia {epoch_train_acc:.2f}%")
    print(f"  ➤ Média da época - Validação: Perda {epoch_val_loss:.4f} | Acurácia {epoch_val_acc:.2f}%")

    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  → Melhor modelo salvo com perda de validação: {best_val_loss:.4f}")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"\n🔴 Early stopping ativado após {epoch+1} épocas.")
            break

    # Atualiza taxa de aprendizado
    scheduler.step(epoch_val_loss)

print(f"\n✅ Treino finalizado em {(time() - inicio)/60:.2f} minutos.")
print(f"✅ Melhor modelo salvo como '{best_model_path}'")

# Plotar gráficos
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(14, 5))

# Gráfico de perda
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Treino', color='blue')
plt.plot(epochs_range, val_losses, label='Validação', color='red', linestyle='--')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico de acurácia
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Treino', color='blue')
plt.plot(epochs_range, val_accuracies, label='Validação', color='red', linestyle='--')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()