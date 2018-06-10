import cv2
import numpy as np
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms


# def flatten(x: torch.Tensor) -> torch.Tensor:
#     N, C, H, W = x.size()  # read in N, C, H, W
#     return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class HandDetectorNet(nn.Module):
    """
    Convolutional neural network to determine whether or not a given image contains a hand.

    Architecture:
    { [convolutional layer] -> [batchnorm] -> [max pool] -> [ReLU] } x 3 -> [affine layer] -> ReLU -> [affine layer] -> [softmax]
    """
    def __init__(self, restore=False, outfile='cache/state.pkl'):
        super().__init__()

        # architecture definition
        self.conv_one = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.batch_norm_one = nn.BatchNorm2d(32)
        self.conv_two = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.batch_norm_two = nn.BatchNorm2d(64)
        self.conv_three = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.batch_norm_three = nn.BatchNorm2d(128)
        self.affine_one = nn.Linear(8 * 8 * 128, 128)
        self.affine_two = nn.Linear(128, 2)

        outfile_path = pathlib.Path(outfile)
        self._outfile_path = outfile_path
        if restore:
            if not outfile_path.is_file():
                raise TypeError('"{0}" is not a valid file'.format(outfile))
            with self._outfile_path.open('rb') as infile:
                self.load(infile)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_one(x)
        x = self.batch_norm_one(x)      # 32 x 64 x 64
        x = F.max_pool2d(x, 2)          # 32 x 32 x 32
        x = F.relu(x)

        x = self.conv_two(x)
        x = self.batch_norm_two(x)      # 64 x 32 x 32
        x = F.max_pool2d(x, 2)          # 64 x 16 x 16
        x = F.relu(x)

        x = self.conv_three(x)
        x = self.batch_norm_three(x)    # 128 x 16 x 16
        x = F.max_pool2d(x, 2)          # 128 x 8 x 8
        x = F.relu(x)

        x = x.view(-1, 8 * 8 * 128)     # flatten; 8 x 8 x 128
        x = self.affine_one(x)
        x = F.relu(x)

        x = self.affine_two(x)

        return F.log_softmax(x, dim=1)


    def save(self) -> None:
        state = self.state_dict()
        with self._outfile_path.open('wb') as outfile:
            torch.save(state, outfile)

    def load(self, infile: open) -> None:
        state = torch.load(infile)
        self.load_state_dict(state)



def load_image(path_to_image: str, dimensions=(32, 32)) -> np.ndarray:
    image = cv2.imread(path_to_image)
    image = cv2.resize(image, dsize=dimensions)
    return np.reshape(image, (3,) + dimensions)


def load_image_directory(training_dir: str, test_dir: str, batch_size: int, dimensions=(32, 32)) -> (data.Dataset, data.DataLoader, data.Dataset, data.DataLoader):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(dimensions),
                                    transforms.ToTensor(),])
                                    # transforms.Normalize(mean, std)])

    training_set = torchvision.datasets.ImageFolder(training_dir, transform=transform)
    training_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=2)

    test_set = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return training_set, training_loader, test_set, test_loader


def train(classifier: nn.Module, loader: data.DataLoader, criterion: nn.modules.loss, optimizer: optim.Optimizer, epochs=1, print_every=2000) -> None:
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_every == print_every - 1:
                print('[{0}, {1}] loss: {2}'.format(epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0


def train_net() -> nn.Module:
    training_set, training_loader, test_set, test_loader = load_image_directory('data/preprocessed', 'data/test', 4, dimensions=(64, 64))

    net = HandDetectorNet()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train(net, training_loader, criterion=criterion, optimizer=optimizer, epochs=5)

    return net


if __name__ == '__main__':
    # net = nn.Sequential(Flatten(),
    #                     nn.Linear(255792, 2),
    #                     nn.ReLU())

    batch_size = 4
    # training_set, training_loader, test_set, test_loader = load_image_directory('data/preprocessed', 'data/test', batch_size)
    net = train_net()

    image = load_image('data/preprocessed/hand/281L.jpg', dimensions=(64, 64))
    image = torch.Tensor(image)
    image = image.unsqueeze(0)
    # print(image.shape)
    # # image = load_image('data/test/other/20160202_080815000_iOS.png', dimensions=(292, 292))
    # image = torch.Tensor([image])
    #
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    ground_truths = ('hand', 'other')
    print(', '.join([ground_truths[predicted[i]] for i in range(1)]))
    #
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