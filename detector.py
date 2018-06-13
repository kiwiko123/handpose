import collections
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid error "OSError: broken data stream when reading image file"
import scipy.special
from dataset import BoundedHandsDataset



class SaveableNet(nn.Module):
    """
    Save/load learned weights from/to pickled output files.
    In derived classes, `__init__` methods must be structured as follows:
        def __init__(self, ...):
            nn.Module.__init__(self, restore=False, outfile='default')
            #
            # implementation...
            #
            SaveableNet.__init__(self)

    `nn.Module.__init__` must be invoked first.
    `SaveableNet.__init__` requires the architecture to be defined before it can run.
    """
    def __init__(self, restore=False, weight_file='default'):
        """
        Initialize a SaveableNet object.
        If `restore=True`, attempts to load `weight_file`'s learned weights.
        If `weight_file='default'`, it will save to './cache/{CLASS_NAME}_state.pkl'.
        """
        if weight_file == 'default':
            weight_file = './cache/{0}_state.pkl'.format(type(self).__name__)
        weight_file_path = pathlib.Path(weight_file)
        self._weight_file_path = weight_file_path
        if restore:
            if not weight_file_path.is_file():
                raise TypeError('"{0}" is not a valid file'.format(weight_file))
            with self._weight_file_path.open('rb') as infile:
                self.load(infile)

    def save(self) -> None:
        state = self.state_dict()
        with self._weight_file_path.open('wb') as outfile:
            torch.save(state, outfile)

    def load(self, infile: open) -> None:
        state = torch.load(infile)
        self.load_state_dict(state)


class YOLOv2Net(SaveableNet):
    """
    Layer         kernel  stride  output shape
    ---------------------------------------------
    Input                             (416, 416, 3)
    1. Convolution    3×3      1      (416, 416, 16)
       MaxPooling     2×2      2      (208, 208, 16)
    2. Convolution    3×3      1      (208, 208, 32)
       MaxPooling     2×2      2      (104, 104, 32)
    3. Convolution    3×3      1      (104, 104, 64)
       MaxPooling     2×2      2      (52, 52, 64)
    4. Convolution    3×3      1      (52, 52, 128)
       MaxPooling     2×2      2      (26, 26, 128)
    5. Convolution    3×3      1      (26, 26, 256)
       MaxPooling     2×2      2      (13, 13, 256)
    6. Convolution    3×3      1      (13, 13, 512)
       MaxPooling     2×2      1      (13, 13, 512)
    7. Convolution    3×3      1      (13, 13, 1024)
    8. Convolution    3×3      1      (13, 13, 1024)
    9. Convolution    1×1      1      (13, 13, 125)
    ---------------------------------------------

    Source:
    http://machinethink.net/blog/object-detection-with-yolo/
    """
    def __init__(self, restore=False, weight_file='default'):
        """
        padding: (kernel_size - 1) // 2
        :param restore:
        :param weight_file:
        """
        nn.Module.__init__(self)
        self.mode = 'test'

        ###
        ### Architecture Definition
        ###
        self.conv_one = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.max_pool_one = nn.MaxPool2d(2, stride=2)
        self.batch_norm_one = nn.BatchNorm2d(16)

        self.conv_two = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.batch_norm_two = nn.BatchNorm2d(32)

        self.conv_three = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.batch_norm_three = nn.BatchNorm2d(64)

        self.conv_four = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.batch_norm_four = nn.BatchNorm2d(128)

        self.conv_five = nn.Conv2d(128, 256, 3, stride=1, padding=2)    # padding=1 ???
        self.batch_norm_five = nn.BatchNorm2d(256)

        self.conv_six = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.max_pool_six = nn.MaxPool2d(2, stride=1, padding=(2-1)//2)
        self.batch_norm_six = nn.BatchNorm2d(512)

        self.conv_seven = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.conv_eight = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.batch_norm_eight = nn.BatchNorm2d(1024)

        self.conv_nine = nn.Conv2d(1024, 35, 1, stride=1)

        self.affine_one = nn.Linear(13 * 13 * 35, 35)
        self.affine_two = nn.Linear(35, 2)

        SaveableNet.__init__(self, restore=restore, weight_file=weight_file)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape (C, H, W)
        # (3, 416, 416)
        x = self.conv_one(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_one(x)
        x = F.leaky_relu(x)

        # (16, 208, 208)
        x = self.conv_two(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_two(x)
        x = F.leaky_relu(x)

        # (32, 104, 104)
        x = self.conv_three(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_three(x)
        x = F.leaky_relu(x)

        # (64, 52, 52)
        x = self.conv_four(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_four(x)
        x = F.leaky_relu(x)

        # (128, 26, 26)
        x = self.conv_five(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_five(x)
        x = F.leaky_relu(x)

        # (256, 14, 14)
        x = self.conv_six(x)
        x = self.max_pool_six(x)
        x = self.batch_norm_six(x)
        x = F.leaky_relu(x)

        # (512, 13 13)
        x = self.conv_seven(x)
        x = self.conv_eight(x)
        x = self.batch_norm_eight(x)
        x = F.leaky_relu(x)

        # (1024, 13, 13)
        x = self.conv_nine(x)

        if self.mode == 'train':
            x = x.view(-1, 13 * 13 * 35)

        # (35, 13, 13)
        return x


def load_image_directory(training_dir: str, test_dir: str, batch_size: int, dimensions: (int, int)) -> (data.Dataset, data.DataLoader, data.Dataset, data.DataLoader):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(dimensions),
                                    transforms.ToTensor(),])
                                    # transforms.Normalize(mean, std)])

    # training_set = torchvision.datasets.ImageFolder(training_dir, transform=transform)
    training_set = BoundedHandsDataset(training_dir, './data/Dataset/annotation.json', batch_size, dimensions, transform=transform)
    training_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=2)

    test_set = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return training_set, training_loader, test_set, test_loader


def normalize_tx(x: float, threshold=1e-10) -> float:
    """
    Normalize a given number so that it can be passed to the logit function.
    The logit is the inverse of the sigmoid function, and only accepts values in the range (0, 1).
    `threshold` is used to determine the closest value within the bounds.

    Returns the normalized x.
    """
    max_threshold = 1 - threshold
    x = max(x, threshold)
    x = min(x, max_threshold)

    return x


def reconstruct_ground_truth_labels(features: torch.Tensor, signed_regions: [[float]], ground_truth_bounding_boxes: (float,), threshold=1e-10) -> torch.Tensor:
    """
    Alters `features` by comparing its incorrect predictions with the ground-truth labels.
    Ensure that `features` is a `clone()` of the net's output.
    Pass this as the second argument to the loss function for learning.

    Returns the updates features tensor.
    """
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    batch_size = features.size(0)
    features = features.data.numpy()

    for batch in range(batch_size):
        true_tl_x, true_tl_y, true_br_x, true_br_y = [t[batch].item() for t in ground_truth_bounding_boxes]
        width = true_br_x - true_tl_x
        height = true_br_y - true_tl_y
        mid_x = true_tl_x + (width / 2)
        mid_y = true_tl_y + (height / 2)

        for cy in range(13):
            for cx in range(13):
                for b in range(5):
                    channel = b * 7

                    # if the "grid" at (cx, cy) does not contain a hand,
                    # assign it a value of -2.5.
                    # Sigmoid(-2.5) ~ 0.07, indicating a very low confidence.
                    if signed_regions[batch, cx, cy] == 0:
                        features[batch, channel + 4, cx, cy] = -10
                        # features[batch, channel + 4, cx, cy] = -2.5

                    # otherwise, the bounding box is around the grid at (cx, cy).
                    else:
                        features[batch, channel + 4, cx, cy] = 7.5
                        # features[batch, channel + 4, cx, cy] = 7.5

                        # the logit function is the inverse of the sigmoid function.
                        # reverse-engineer the calculations done in `bounding_box()` based on the ground-truth coordinates.
                        tx = mid_x / 32
                        tx -= cx
                        tx = normalize_tx(tx, threshold=threshold)
                        features[batch, channel + 0, cx, cy] = scipy.special.logit(tx)

                        ty = mid_y / 32
                        ty -= cy
                        ty = normalize_tx(ty, threshold=threshold)
                        features[batch, channel + 1, cx, cy] = scipy.special.logit(ty)

                        # similarly, the natural logarithm is the inverse of the exponential function,
                        # which is applied to `tw` and `th` in `bounding_box()`.
                        tw = width / 32
                        tw /= anchors[2 * b]
                        features[batch, channel + 2, cx, cy] = np.log(tw)

                        th = height / 32
                        th /= anchors[2 * b + 1]
                        features[batch, channel + 3, cx, cy] = np.log(th)

    return torch.Tensor(features)


def bounding_box(outputs: torch.Tensor) -> [(float, (int,))]:
    """
    Computes the predicted confidence score and bounding box coordinates from `outputs`.
    `outputs` is the Tensor predicted by a net.

    Results are returned as a list (deque) of tuples, where each tuple is in the format:
      `(confidence_score, (top_left_x, top_left_y, bottom_right_x, bottom_right_y))`
    where the inner tuple represents the bounding box coordinates.

    The most confident score is maintained throughout the calculations,
    and will always be the first element in the returned list.

    Source:
    http://machinethink.net/blog/object-detection-with-yolo/
    """
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    predictions = collections.deque()
    most_accurate = ()
    batches = outputs.size(0)
    for batch in range(batches):
        features = outputs[batch]
        # features = features.permute(0, 1, 2)    # reshape to (35x13x13)
        for cy in range(13):
            for cx in range(13):
                for b in range(5):
                    channel = b * 7
                    tx = features[channel + 0, cx, cy]
                    ty = features[channel + 1, cx, cy]
                    tw = features[channel + 2, cx, cy]
                    th = features[channel + 3, cx, cy]
                    tc = features[channel + 4, cx, cy]

                    x = (cx + F.sigmoid(tx)) * 32
                    y = (cy + F.sigmoid(ty)) * 32
                    w = np.exp(tw.item()) * anchors[2 * b] * 32
                    h = np.exp(th.item()) * anchors[2 * b + 1] * 32
                    c = F.sigmoid(tc)

                    x, y, c = x.item(), y.item(), c.item()
                    classes = torch.Tensor([features[channel + 5 + i, cx, cy] for i in range(2)])
                    classes = F.softmax(classes, dim=0)
                    best_score, prediction = [t.item() for t in torch.max(classes, dim=0)]
                    confidence = best_score * c

                    if confidence > 0:
                        x -= w / 2
                        y -= h / 2

                        if 0 <= x < 416 and 0 <= y < 416:
                            entry = (confidence, (x, y, x + w, y + h))
                            if most_accurate:
                                most_accurate = max(most_accurate, entry, key=lambda e: e[0])
                            else:
                                most_accurate = entry
                            predictions.append(entry)
                            predictions.append((confidence, (x, y, x + w, y + h)))

    if most_accurate:
        predictions.appendleft(most_accurate)

    return predictions


def train(classifier: nn.Module, loader: data.DataLoader, criterion: nn.modules.loss, optimizer: optim.Optimizer, epochs=1, print_every=-1) -> None:
    """
    Performs the actual training of `classifier`.
    This is a generic function that will work with any PyTorch-compatible parameters.

    `criterion` is the loss function.
    """
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            images = data['image']
            bounding_box_coords = data['bounding_box']
            signed_regions = data['signed_regions']

            outputs = classifier(images)
            features = outputs.clone()
            features = features.view(-1, 35, 13, 13)
            labels = reconstruct_ground_truth_labels(features, signed_regions, bounding_box_coords)
            labels = labels.view(-1, 35 * 13 * 13)

            optimizer.zero_grad()
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if print_every > 0 and i % print_every == print_every - 1:
                print('[{0}, {1}] loss: {2}'.format(epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0



def train_net(net: SaveableNet, batch_size: int, epochs=10, dimensions=(416, 416)) -> SaveableNet:
    """
    Quick function to train a new net.
    Assesses loss through mean-squared error (MSE).
    Optimizes through stochastic gradient descent (SGD),
    which we determined performs much better than Adam.

    Returns `net`, trained.
    """
    training_set, training_loader, test_set, test_loader = load_image_directory('data/preprocessed_hw3/train', 'data/preprocessed_hw3/test', batch_size=batch_size, dimensions=dimensions)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=1e-5)
    train(net, training_loader, criterion=criterion, optimizer=optimizer, epochs=epochs, print_every=1)

    return net



if __name__ == '__main__':
    batch_size = 16
    epochs = 8
    dimensions = (416, 416)
    model = YOLOv2Net()
    model.mode = 'train'
    net = train_net(model, batch_size, epochs=epochs, dimensions=dimensions)
    net.save()

    # net = YOLOv2Net(restore=True)
    net.mode = 'test'

    image_path = './data/Dataset/Color/022_1757.jpg'
    # image_path = '/Users/geoffreyko/OneDrive/Pictures/Camera Roll/20160729_022841530_iOS.jpg'
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=dimensions)
    image = np.reshape(image, (3,) + dimensions)
    image = torch.Tensor(image)
    image.unsqueeze_(0)

    outputs = net(image)
    print(outputs)
    p = bounding_box(outputs)

    for confidence, coordinates in sorted(p, key=lambda e: e[0], reverse=True):
    # confidence, coordinates = max(p, key=lambda e: e[0])
        color = cv2.imread(image_path)
        color = cv2.resize(color, dsize=dimensions)
        tl_x, tl_y, br_x, br_y = [int(c) for c in coordinates]
        green = (0, 255, 0)
        cv2.rectangle(color, (tl_x, tl_y), (br_x, br_y), green, 1)
        cv2.imshow('frame', color)
        print(confidence)
        cv2.waitKey(0)
