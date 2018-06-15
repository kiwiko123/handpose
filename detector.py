import collections
import cv2
import math
import numpy as np
import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import scipy.special
from PIL import ImageFile
from torchvision import transforms
from dataset import Assignment3Dataset, HAND, OTHER

ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid error "OSError: broken data stream when reading image file"
ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]


class Prediction:
    """
    Convenience class to encompass prediction data.
    """
    def __init__(self, confidence: float, bounding_box: (int,) * 4, prediction: int, grid_x: int, grid_y: int):
        self.confidence = confidence
        self.bounding_box = bounding_box
        self.prediction = prediction
        self.grid_x = grid_x
        self.grid_y = grid_y

    def __iter__(self):
        yield (self.confidence, self.bounding_box)



class SaveableNet(nn.Module):
    """
    Save/load learned weights to/from pickled output files.
    In derived classes, `__init__` methods must be structured as follows:

    1) `nn.Module.__init__` must be invoked first.
    2) `SaveableNet.__init__` requires the architecture to be defined before it can run.

    For example:

        def __init__(self, restore=False, weight_file='default'):
            nn.Module.__init__(self, restore=restore, weight_file=weight_file)
            #
            # implementation...
            #
            SaveableNet.__init__(self)
    """
    def __init__(self, restore=False, weight_file='default'):
        """
        Initialize a SaveableNet object.
        If `restore=True`, attempts to load `weight_file`'s learned weights.
        If `weight_file='default'`, it will save to './cache/{CLASS_NAME}_state.pkl'.

        If the directory specified by weight_file does not exist, it will be created automatically.
        """
        if weight_file == 'default':
            weight_file = './cache/{0}_state.pkl'.format(type(self).__name__)

        weight_file_path = pathlib.Path(weight_file)
        if not weight_file_path.parent.is_dir():
            os.makedirs(str(weight_file_path.parent))

        self.restore = restore
        self._weight_file_path = weight_file_path
        if restore:
            self.load()


    def save(self, to='default') -> None:
        """
        Saves the learned weights into a pickled output file, specified by `to`.
        If `to='default'`, then the file specified in __init__ is used.
        """
        out = self._weight_file_path if to == 'default' else pathlib.Path(to)
        if not out.is_file():
            raise ValueError('{0} is not a valid file'.format(out))

        state = self.state_dict()
        with out.open('wb') as outfile:
            torch.save(state, outfile)


    def load(self, from_='default') -> None:
        """
        Loads the learned weights from the file pointed to by 'from_'.
        Default behavior is the same as in method `save()`.
        """
        in_ = self._weight_file_path if from_ == 'default' else pathlib.Path(from_)

        with in_.open('rb') as infile:
            state = torch.load(infile)
            self.load_state_dict(state)



class YOLOv2Net(SaveableNet):
    """
    9 layer convolutional neural network aimed at object detection.

    Architecture
    ------------
    | Layer       | kernel | stride | output_shape |
    ---------------------------------------------
    Input                             (3, 416, 416)
    1. convolution    3×3      1      (16, 416, 416)
       max pool       2×2      2      (16, 208, 208)
       batch norm                     (16, 208, 208)
       leaky ReLU                     (16, 208, 208)
    2. convolution    3×3      1      (32, 208, 208)
       max pool       2×2      2      (32, 104, 104)
       batch norm                     (32, 104, 104)
       leaky ReLU                     (32, 104, 104)
    3. convolution    3×3      1      (64, 104, 104)
       max pool       2×2      2      (64, 52, 52)
       batch norm                     (64, 52, 52)
       leaky ReLU                     (64, 52, 52)
    4. convolution    3×3      1      (128, 52, 52)
       max pool       2×2      2      (128, 26, 26)
       batch norm                     (128, 26, 26)
       leaky ReLU                     (128, 26, 26)
    5. convolution    3×3      1      (256, 26, 26)
       max pool       2×2      2      (256, 14, 14)
       batch norm                     (256, 14, 14)
       leaky ReLU                     (256, 14, 14)
    6. convolution    3×3      1      (512, 14, 14)
       max pool       2×2      1      (512, 13, 13)
       batch norm                     (512, 13, 13)
       leaky ReLU                     (512, 13, 13)
    7. convolution    3×3      1      (1024, 13, 13)
       batch norm                     (1024, 13, 13)
       leaky ReLU                     (1024, 13, 13)
    8. convolution    3×3      1      (13, 13, 1024)
       batch norm                     (1024, 13, 13)
       leaky ReLU                     (1024, 13, 13)
    9. convolution    1×1      1      (35, 13, 13)
    ---------------------------------------------

    Source:
    http://machinethink.net/blog/object-detection-with-yolo/
    """
    def __init__(self, restore=False, weight_file='default'):
        """
        padding: (kernel_size - 1) // 2
        """
        nn.Module.__init__(self)
        self.mode = 'test'
        self._leaky_relu_slope = 0.1
        in_channels = 1  # 1 for grayscale, 3 for RGB

        self.conv_one = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1)
        self.max_pool_one = nn.MaxPool2d(2, stride=2)
        self.batch_norm_one = nn.BatchNorm2d(16)

        self.conv_two = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.batch_norm_two = nn.BatchNorm2d(32)

        self.conv_three = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.batch_norm_three = nn.BatchNorm2d(64)

        self.conv_four = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.batch_norm_four = nn.BatchNorm2d(128)

        self.conv_five = nn.Conv2d(128, 256, 3, stride=1, padding=2)
        self.batch_norm_five = nn.BatchNorm2d(256)

        self.conv_six = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.max_pool_six = nn.MaxPool2d(2, stride=1, padding=(2-1)//2)
        self.batch_norm_six = nn.BatchNorm2d(512)

        self.conv_seven = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.batch_norm_seven = nn.BatchNorm2d(1024)

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
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

        # (16, 208, 208)
        x = self.conv_two(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_two(x)
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

        # (32, 104, 104)
        x = self.conv_three(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_three(x)
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

        # (64, 52, 52)
        x = self.conv_four(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_four(x)
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

        # (128, 26, 26)
        x = self.conv_five(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_five(x)
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

        # (256, 14, 14)
        x = self.conv_six(x)
        x = self.max_pool_six(x)
        x = self.batch_norm_six(x)
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

        # (512, 13 13)
        x = self.conv_seven(x)
        x = self.batch_norm_seven(x)
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

        # (1024, 13, 13)
        x = self.conv_eight(x)
        x = self.batch_norm_eight(x)
        x = F.leaky_relu(x, negative_slope=self._leaky_relu_slope)

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
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),])
                                    # transforms.Normalize(mean, std)])

    training_set = Assignment3Dataset(training_dir, './data/Dataset/annotation.json', batch_size, dimensions, transform=transform)
    training_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = Assignment3Dataset(test_dir, './data/Dataset/annotation.json', batch_size, dimensions, transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

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


def reconstruct_ground_truth_labels(features: torch.Tensor, signed_regions: [[float]], ground_truth_bounding_boxes: (float,), ground_truth_classes: torch.Tensor, threshold=1e-10) -> torch.Tensor:
    """
    Alters `features` by comparing its incorrect predictions with the ground-truth labels.
    Ensure that `features` is a `clone()` of the net's output.
    Pass this as the second argument to the loss function for learning.

    Returns the updates features tensor.
    """
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
                    # sigmoid(-2.5) ~ 0.07, indicating a very low confidence.
                    if ground_truth_classes[batch] == OTHER or signed_regions[batch, cx, cy] == 0:
                        # features[batch, channel + 4, cx, cy] = -10
                        features[batch, channel + 4, cx, cy] = -2.5
                        features[batch, channel + 5 + HAND] = -1.1
                        features[batch, channel + 5 + OTHER] = 1.1

                    # otherwise, the bounding box is around the grid at (cx, cy).
                    else:
                        # this (tc) represents the confidence score that some object is within the bounding box.
                        # 1.1 is chosen because sigmoid(1.1) ~ 0.75
                        features[batch, channel + 4, cx, cy] = 1.1

                        # the logit function is the inverse of the sigmoid function.
                        # reverse-engineer the calculations done in `bounding_box()` based on the ground-truth coordinates.
                        tx = mid_x / 32
                        tx -= cx
                        tx = normalize_tx(tx, threshold=threshold)
                        tx = scipy.special.logit(tx)
                        features[batch, channel, cx, cy] = tx

                        ty = mid_y / 32
                        ty -= cy
                        ty = normalize_tx(ty, threshold=threshold)
                        ty = scipy.special.logit(ty)
                        features[batch, channel + 1, cx, cy] = ty

                        if any(math.isnan(i) or i == math.inf for i in (tx, ty)):
                            raise ValueError

                        # similarly, the natural logarithm is the inverse of the exponential function,
                        # which is applied to `tw` and `th` in `bounding_box()`.
                        tw = width / 32
                        tw /= ANCHORS[2 * b]
                        features[batch, channel + 2, cx, cy] = np.log(tw)

                        th = height / 32
                        th /= ANCHORS[2 * b + 1]
                        features[batch, channel + 3, cx, cy] = np.log(th)
                        features[batch, channel + 5 + HAND] = 1.1
                        features[batch, channel + 5 + OTHER] = -1.1

    return torch.Tensor(features)


def bounding_box(outputs: torch.Tensor) -> [Prediction]:
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

                    x = (cx + F.sigmoid(tx) + 1) * 32
                    y = (cy + F.sigmoid(ty) + 1) * 32
                    w = np.exp(tw.item()) * ANCHORS[2 * b] * 32
                    h = np.exp(th.item()) * ANCHORS[2 * b + 1] * 32
                    c = F.sigmoid(tc)

                    x, y, c = x.item(), y.item(), c.item()
                    classes = torch.Tensor([features[channel + 5 + i, cx, cy] for i in range(2)])
                    classes = F.softmax(classes, dim=0)
                    best_score, prediction = [t.item() for t in torch.max(classes, dim=0)]
                    confidence = best_score * c

                    # if prediction == OTHER:
                    #     confidence /= 10

                    # if prediction == HAND: # confidence > 0:
                    x -= w / 2
                    y -= h / 2

                    if 0 <= x < 416 and 0 <= y < 416:
                        entry = Prediction(confidence, (x, y, x + w, y + h), prediction, cx, cy)
                        if most_accurate:
                            most_accurate = max(most_accurate, entry, key=lambda e: (e.confidence))
                        else:
                            most_accurate = entry
                        predictions.append(entry)

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
            ground_truth_class = data['class']

            outputs = classifier(images)
            features = outputs.clone()
            features = features.view(-1, 35, 13, 13)
            labels = reconstruct_ground_truth_labels(features, signed_regions, bounding_box_coords, ground_truth_class)
            labels = labels.view(-1, 35 * 13 * 13)

            optimizer.zero_grad()
            loss = criterion(outputs.float(), labels)
            if loss == math.inf:
                break
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if print_every > 0 and i % print_every == print_every - 1:
                print('[{0}, {1}] loss: {2}'.format(epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0


def validate(classifier: nn.Module, loader: data.DataLoader, batch_size: int) -> None:
    """
    Validation of the classifier. Pass in the test loader as the second argument.
    Images will display to the screen - press any button to continue.
    """
    for batch in range(batch_size):
        for i, data in enumerate(loader):
            image = data['image']
            outputs = classifier(image)
            predictions = bounding_box(outputs)
            best_prediction = predictions[0]

            coordinates = best_prediction.bounding_box
            color = image.data.numpy()
            n, c, b, w = color.shape
            color = np.reshape(color, (n, b, w, c))
            color = color[batch]
            tl_x, tl_y, br_x, br_y = [int(c) for c in coordinates]
            red = (0, 0, 255)
            cv2.rectangle(color, (tl_x, tl_y), (br_x, br_y), red, 1)
            print('Predicted coordinates:', (tl_x, tl_y), (br_x, br_y))
            cv2.imshow('frame', color)
            cv2.waitKey(0)



if __name__ == '__main__':
    ###
    ### Hyperparameters
    ###
    batch_size = 16
    epochs = 8
    dimensions = (416, 416)
    learning_rate = 1e-3
    training_dir = './data/preprocessed_hw3/train_with_other'
    test_dir = './data/preprocessed_hw3/test/hand'

    training_set, training_loader, test_set, test_loader = load_image_directory(training_dir, test_dir,
                                                                                batch_size=batch_size, dimensions=dimensions)

    ###
    ### Set up the model
    ###
    model = YOLOv2Net()
    model.mode = 'train'

    ###
    ### Loss function, and optimizer
    ###
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=1e-6)

    ###
    ### Train, and save weights
    ###

    # train(model, training_loader, criterion=criterion, optimizer=optimizer, epochs=epochs, print_every=1)
    model = YOLOv2Net(restore=True)
    if not model.restore:
        model.save()
    model.mode = 'test'


    ###
    ### User-study test
    ###
    image_path = './data/Dataset/Color/083_2567.jpg'

    in_channels = 1
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, dsize=dimensions)
    image = np.reshape(image, (in_channels,) + dimensions)
    image = torch.Tensor(image)
    image.unsqueeze_(0)

    outputs = model(image)
    predictions = bounding_box(outputs)

    for entry in sorted(p, key=lambda e: e.confidence, reverse=True):
        confidence = entry.confidence
        coordinates = entry.bounding_box
        color = cv2.imread(image_path)
        color = cv2.resize(color, dsize=dimensions)
        tl_x, tl_y, br_x, br_y = [int(c) for c in coordinates]
        green = (0, 255, 0)
        cv2.rectangle(color, (tl_x, tl_y), (br_x, br_y), green, 1)
        cv2.imshow('frame', color)
        print(entry.prediction, confidence)
        cv2.waitKey(0)


    ###
    ### Validation
    ###
    validate(model, test_loader, batch_size)