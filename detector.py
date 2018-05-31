# Create a model to predict if a given image contains a hand or not.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
from PIL import Image


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class ConvolutionalNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def load_image(path_to_image: str, dimensions=(32, 32)) -> np.ndarray:
    image = cv2.imread(path_to_image)
    image = cv2.resize(image, dsize=dimensions)
    return np.reshape(image, (3,) + dimensions)


def load_image_directory(training_dir: str, test_dir: str):
    mean = [0.5, 0.5, 0.5]
    std = mean
    transform = transforms.Compose([transforms.Resize((292, 292)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    training_set = torchvision.datasets.ImageFolder(training_dir, transform=transform)
    training_loader = data.DataLoader(training_set, batch_size=1, shuffle=False, num_workers=2)

    test_set = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    ground_truths = ('hand', 'other')

    return training_set, training_loader, test_set, test_loader


def load_cifar10() -> ('DataLoader',):
    mean = [0.5, 0.5, 0.5]
    std = mean
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return training_set, training_loader, test_set, test_loader


def train(classifier: nn.Module, loader, criterion: nn.modules.loss, optimizer: optim, epochs=1, print_every=2000):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every - 1:
                print('[{0}, {1}] loss: {2}'.format(epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def train_net():
    training_set, training_loader, test_set, test_loader = load_image_directory('data/test', 'data/test')

    # net = ConvolutionalNet()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net = nn.Sequential(Flatten(),
                        nn.Linear(255792, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, training_loader, criterion=criterion, optimizer=optimizer, epochs=1)

    return net


if __name__ == '__main__':
    # training_set, training_loader, test_set, test_loader = load_cifar10()
    training_set, training_loader, test_set, test_loader = load_image_directory('data/test', 'data/test')

    # net = ConvolutionalNet()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net = nn.Sequential(Flatten(),
                        nn.Linear(255792, 2),
                        nn.ReLU())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train(net, training_loader, criterion=criterion, optimizer=optimizer, epochs=1)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    print(labels)

    image = load_image('data/test/hand/281L.jpg', dimensions=(292, 292))
    print(image.shape)
    # image = load_image('data/test/other/20160202_080815000_iOS.png', dimensions=(292, 292))
    image = torch.Tensor([image])

    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    print(predicted)

    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                               for j in range(4)))

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: {0}%'.format(100 * correct / total))