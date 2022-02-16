import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from tqdm import tqdm

from model import ResNet50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 1000
SAVE_EVERY_X_EPOCHS = 1
SAVE_MODEL_LOC = "./save_"
LOAD_MODEL_LOC = "./example_parameters"


# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, train_loader, validation_loader, num_classes=10):
    accuracies = []
    total_losses = []
    for epoch in range(NUMBER_OF_EPOCHS):
        # Training
        model.train()
        progress = tqdm(train_loader)

        # get the inputs; data is a list of [x, y]
        for i, (x, y) in enumerate(progress):
            # https://stackoverflow.com/a/58677827
            x = nnf.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            x_ = model(x)
            loss = loss_function(x_, y)

            # make the progress bar display loss
            progress.set_postfix(loss=loss.item())

            # backpropagation
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()

        progress = tqdm(validation_loader)

        total_correct = 0
        total_loss = 0.0
        total = 0
        cm = np.zeros((num_classes, num_classes))

        for i, (x, y) in enumerate(progress):
            x = nnf.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            x_ = model(x)

            batch_correct = torch.sum(torch.argmax(x_, dim=1) == y).item()
            total_correct += batch_correct

            cm[y][torch.argmax(x_, dim=1)] += 1

            total += x_.shape[0]

            loss = loss_function(x_, y)
            total_loss += loss.item()
        accuracy = (total_correct * 100.0) / total
        total_loss /= total
        print("Validation accuracy = " + str(accuracy) + "% = " + str(total_correct) + "/" + str(total))
        print("Validation avg. loss = " + str(total_loss))
        accuracies.append(accuracy)
        total_losses.append(total_loss)

        if SAVE_MODEL_LOC:
            torch.save(model.state_dict(), SAVE_MODEL_LOC + str(epoch))

        plt.figure(figsize=(10, 10), dpi=100)
        plt.scatter(range(0, epoch + 1), accuracies)
        plt.title("accuracy")
        plt.show()
        
        plt.figure(figsize=(10, 10), dpi=100)
        plt.scatter(range(0, epoch + 1), total_losses)
        plt.title("avg loss")
        plt.show()

        class_labels = list(range(num_classes))
        ax = sn.heatmap(
            cm,
            annot=True,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            fmt='g')
        ax.set(
            xlabel="prediction",
            ylabel="truth",
            title="Validation Confusion Matrix")
        plt.show()


if __name__ == "__main__":
    # create/load model
    model = ResNet50(3, 10).to(DEVICE)

    if LOAD_MODEL_LOC:
        model.load_state_dict(torch.load(LOAD_MODEL_LOC))

    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # used defaults from pytorch tutorial as it was https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    train(model, loss_function, optimizer, trainloader, testloader)
