import torchvision
import torch
import time
import numpy as np

from torchvision import datasets, transforms
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Configurar device (CPU o GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformaciones para train y test
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar datasets
    train_dir = 'train'
    test_dir = 'test'
    train_dataset = datasets.ImageFolder(train_dir, transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=0)

    # Preparar el modelo
    model = torchvision.models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    num_epochs = 2
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} running")
        model.train()
        running_loss = 0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100. * running_corrects / len(train_dataset)

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        print('[Train] Loss: {:.4f} Acc: {:.2f}%'.format(epoch_loss, epoch_acc))

        # ---- Validación después del epoch ----
        model.eval()
        running_loss = 0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = 100. * running_corrects / len(test_dataset)

        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_acc)

        print('[Test] Loss: {:.4f} Acc: {:.2f}%'.format(epoch_loss, epoch_acc))

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'resnet18_finetuned.pth')
    print("\nModelo guardado como resnet18_finetuned.pth 🧠💾")

    print("\nEntrenamiento finalizado en {:.2f} segundos".format(time.time() - start_time))
