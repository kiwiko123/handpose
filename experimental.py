import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class HandDetectorNet(nn.Module):
    """
    Convolutional neural network to determine whether or not a given image contains a hand.

    Architecture:
    { [convolutional layer] -> [batchnorm] -> [max pool] -> [ReLU] } x 3 -> [affine layer] -> ReLU -> [affine layer] -> [softmax]

    Source: https://cs230-stanford.github.io/pytorch-vision.html
    """
    def __init__(self):
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


class EgohandsNet(nn.Module):
    """
    Source: http://vision.soic.indiana.edu/egohands_files/hand_classifier.prototxt
    """
    def __init__(self):
        super().__init__(restore=restore, outfile=outfile)

        # architecture
        self.conv_one = nn.Conv2d(3, 96, 11, stride=4)
        self.max_pool_one = nn.MaxPool2d(3, stride=2)
        self.batch_norm_one = nn.BatchNorm2d(96)
        self.conv_two = nn.Conv2d(96, 256, 5, padding=2)
        self.max_pool_two = nn.MaxPool2d(3, stride=2)
        self.batch_norm_two = nn.BatchNorm2d(256)
        self.conv_three = nn.Conv2d(256, 384, 3, padding=1, bias=False)
        self.conv_four = nn.Conv2d(384, 384, 3, padding=1)
        self.conv_five = nn.Conv2d(384, 256, 3, padding=1)
        # self.max_pool_three = nn.MaxPool2d(8, stride=2)
        self.max_pool_three = nn.AdaptiveMaxPool2d(8)
        self.affine_one = nn.Linear(8 * 8 * 256, 4096)
        self.dropout_ratio = 0.5
        self.affine_two = nn.Linear(4096, 4096)
        self.affine_three = nn.Linear(4096, 2, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_one(x)
        x = F.relu(x)
        x = self.max_pool_one(x)
        x = self.batch_norm_one(x)

        x = self.conv_two(x)
        x = F.relu(x)
        x = self.batch_norm_two(x)

        x = self.conv_three(x)
        x = F.relu(x)

        x = self.conv_four(x)
        x = F.relu(x)

        x = self.conv_five(x)
        x = F.relu(x)
        x = self.max_pool_three(x)

        x = x.view(-1, 8 * 8 * 256)
        x = self.affine_one(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.affine_two(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.affine_three(x)

        return F.log_softmax(x, dim=1)



###
### Functions for capturing live webcam, with background subtraction.
###
def capture_background_image(capturer: cv2.VideoCapture, dimensions=(64, 64), skip=15) -> np.ndarray:
    """
    Captures and returns a single, still image from the webcam to use as the background.
    Use this image as the initial training point for a cv2.BackgroundSubtractorMOG2 object.
    `skip` is the number of frames to skip before capturing the image, to allow for pre-focusing.
    """
    for i in range(skip):
        capturer.read()
    ret, frame = capturer.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, dsize=dimensions)
    return image


def normalize_image_for_net(image: np.ndarray, background_subtractor: cv2.BackgroundSubtractorMOG2, dimensions=(64, 64)) -> torch.Tensor:
    """
    Prepares `image` to be passed to an `nn.Module`-like object.
    `image` is a numpy array that can be obtained through an OpenCV operation (i.e., `cv2.imread`).
    `background_subtractor` is the BackgroundSubtractorMOG2 object that has already been trained on a single background image.
    `dimensions` specify how much to resize `image`.
    Ensures `image` has 3 channels, and converts it to a Tensor object.
    Returns the Tensor.
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(result, dsize=dimensions)
    result = background_subtractor.apply(result)
    result = np.stack((result,) * 3, -1)  # make image have 3 channels (necessary for 1st convolutional layer)
    result = np.reshape(result, (3,) + dimensions)
    result = torch.Tensor(result)
    result.unsqueeze_(0)

    return result


def track_background_subtract(predict_every=5) -> None:
    """
    Uses the webcam to track live video.
    Predicts whether or not a hand is detected in each frame.
    If a hand is predicted, prints "hand" to the console, or "other" if not.
    `predict_every` determines the number of frames to skip before making a prediction.

    Press 'q' to end webcam capture.
    """
    ground_truths = ('hand', 'other')
    dimensions = (64, 64)
    i = 1
    net = create_net()
    capturer = cv2.VideoCapture(0)
    model = create_background_subtractor()
    background = capture_background_image(capturer)
    model.apply(background)

    while capturer.isOpened():
        ret, frame = capturer.read()

        if i % predict_every == 0:
            image = normalize_image_for_net(frame, model, dimensions=dimensions)
            outputs = net(image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted[0].item()
            print(ground_truths[prediction])

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    capturer.release()
    cv2.destroyAllWindows()
    print()
    prompt_for_save(net)


def _visualize_background_subtraction(**background_subtractor_kwargs) -> None:
    """
    Helper function to visualize how background subtraction works.
    Tweak `background_subtractor_kwargs` to see how things are changed.
    """
    capturer = cv2.VideoCapture(0)
    background_subtractor = create_background_subtractor(**background_subtractor_kwargs)
    background = capture_background_image(capturer)
    background_subtractor.apply(background)

    while capturer.isOpened():
        ret, frame = capturer.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = background_subtractor.apply(image)

        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capturer.release()
    cv2.destroyAllWindows()