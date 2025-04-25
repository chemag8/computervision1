import torch
from torchvision.datasets import ImageFolder # lee carpetas, asignando etiquetas a las imagenes segun la carpeta en la que se encuentran
from torchvision import transforms # para transformar las imagenes
from torch.utils.data import DataLoader # para cargar los datos en batches
import torchvision.models as models # modelos preentrenados de pytorch
import torch.nn as nn # para definir la red neuronal
import torch.optim as optim # para optimizar la red neuronal

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
  # Transformaciones para el conjunto de entrenamiento (con data augmentation)
  train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(224), # recorta la imagen a un tamaño aleatorio entre 224x224 y 256x256
      transforms.RandomHorizontalFlip(), # voltea la imagen horizontalmente con una probabilidad del 50%
      transforms.ToTensor() 
  ])
  
  # Transformaciones para el conjunto de test
  test_transforms = transforms.Compose([
      transforms.Resize(256), # redimensiona la imagen a 256x256
      transforms.CenterCrop(224), # recorta la imagen al centro a 224x224
      transforms.ToTensor()
  ])

  train_data = ImageFolder(root=r"train", transform=train_transforms)
  test_data = ImageFolder(root=r"test", transform=test_transforms)
  
  print(train_data.class_to_idx)
  print(test_data.class_to_idx)

  train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # 32 imagenes por batch, mezcla aleatoriamente
  test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

  # Cargar modelo ResNet18 preentrenado
  model = models.resnet18(pretrained=True) # cargar el modelo preentrenado, la arquitectura es ResNet18 e incluye una capa final de 1000 neuronas
  
  # Congelo todos los pesos, solo se entrenará la capa que se añadirá al final
  for param in model.parameters():
      param.requires_grad = False
  
  # Reemplazar la última capa completamente conectada
  num_features = model.fc.in_features # Número de características de entrada de la última capa
  model.fc = nn.Linear(num_features, 2)  # model.fc es la última capa, 2 es el número de clases (benigno y maligno)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Usando dispositivo: {device}")
  
  model = model.to(device)

  # Función de pérdida: clasificación multiclase (aunque solo tengas 2)
  criterion = nn.CrossEntropyLoss() 
  
  # Optimizador: solo entrenamos la capa final
  optimizer = optim.Adam(model.fc.parameters(), lr=0.0001) 
  
  # Numero de épocas
  NUM_EPOCHS = 50

  start_time = time.time()

  train_loss, val_loss, train_acc, val_acc = train_model(
      model, train_loader, test_loader, criterion, optimizer, device)

  #Grafica de Loss
  plt.figure(figsize=(10,5))
  plt.plot(train_loss, label='Train Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.xlabel('Épocas')
  plt.ylabel('Pérdida')
  plt.title('Evolución de la Pérdida')
  plt.legend()
  plt.grid(True)
  plt.savefig('log_loss_curve.png')
  print("Gráfico con escala log y ticks decimales guardado como log_loss_curve.png")

    #Grafica de Accuracy
  plt.figure(figsize=(10,5))
  plt.plot(train_acc, label='Train Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.xlabel('Épocas')
  plt.ylabel('Precisión')
  plt.title('Evolución de la Precisión')
  plt.legend()
  plt.grid(True)
  plt.savefig('accuracy_curve.png')
  print("Gráfico de precisión guardado como accuracy_curve.png")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train() # Establece el modelo en modo de entrenamiento
        running_loss = 0.0 
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Reinicia los gradientes
            outputs = model(inputs) # Propagación hacia adelante
            loss = criterion(outputs, labels)

            loss.backward() # Propagación hacia atrás
            optimizer.step() # Actualiza los pesos

            running_loss += loss.item() * inputs.size(0) # Acumula la pérdida
            _, preds = torch.max(outputs, 1) # Obtiene las predicciones, el índice de la clase con mayor probabilidad
            running_corrects += torch.sum(preds == labels.data) # Acumula el número de aciertos del lote

        epoch_loss = running_loss / len(train_loader.dataset) # Pérdida media por época, dividiendo por el número total de imágenes
        epoch_acc = running_corrects.double() / len(train_loader.dataset) # Precisión media por época

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        # Evaluación en validación
        model.eval() # Establece el modelo en modo de evaluación
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad(): # Desactiva los gradientes para la validación y haces mismos calculos que en el entrenamiento menos la propagación hacia atrás
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc.item())

        print(f"Época {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history



