import torchvision
import torch
import time
import numpy as np

from torchvision import datasets, transforms
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Create a transform function
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

    train_dir = 'train'
    test_dir = 'test'
    train_dataset = datasets.ImageFolder(train_dir, transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=0)

    model = torchvision.models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    print(num_features)

    model.fc = torch.nn.Linear(num_features, 2)
    model = model.to('cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    num_epochs = 2
    start_Time = time.time()

    for epoch in range(num_epochs):
        print("Epoch {} running".format(epoch))
        model.train()
        running_loss = 0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects / len(train_dataset) + 100.

            train_loss.append(epoch_loss)
            train_accuracy.append(epoch_acc)

            # Print progress
            print('[Train # {}] Loss: {:.4f} Acc: {:.4f}'.format(
                i, epoch_loss, epoch_acc))

            model.eval()
            with torch.no_grad():
                running_loss = 0
                running_corrects = 0

                for inputs, labels in test_dataloader:
                    inputs = inputs.to('cpu')
                    labels = labels.to('cpu')

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data).item()

                epoch_loss = running_loss / len(test_dataset)
                epoch_acc = running_corrects / len(test_dataset) + 100.

                test_loss.append(epoch_loss)
                test_accuracy.append(epoch_acc)

                print('[Test # {}] Loss: {:.4f} Acc: {:.4f}'.format(
                    i, epoch_loss, epoch_acc)) 
