import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from model import ResNet50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
NUMBER_OF_EPOCHS = 100
SAVE_EVERY_X_EPOCHS = 1
SAVE_MODEL_LOC = "./model_"
LOAD_MODEL_LOC = None

# a training loop that runs a number of training epochs on a model
def train(model, loss_function, optimizer, train_loader, validation_loader):
    model_metric_scores = {"accuracy": [], "f1": []}

    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        progress = tqdm(train_loader)

        running_loss = 0.0
        # get the inputs; data is a list of [x, y]
        for i, (x, y) in enumerate(progress):
            # https://stackoverflow.com/a/58677827
            x = nnf.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

            # plt.figure(figsize=(10, 10), dpi=100)
            # plt.imshow(x.detach().squeeze().numpy().T)
            # plt.title(i)
            # plt.show()

            x = x.to(DEVICE)
            print(x.dtype)

            y = y.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            x_ = model(x)
            print("x_ = " + str(x_.detach().cpu().numpy()) + ", y = " + str(y.detach().cpu().numpy()))
            loss = loss_function(x_, y)

            # make the progress bar display loss
            progress.set_postfix(loss=loss.item())

            # backpropagation
            loss.backward()
            optimizer.step()

        model.eval()


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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    train(model, loss_function, optimizer, trainloader, testloader)
