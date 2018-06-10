import cv2
import numpy as np
import torch
from detector import train_net, HandDetectorNet
from preprocess import create_background_subtractor


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


def create_net(weight_file='./cache/state.pkl') -> HandDetectorNet:
    """
    Prompts user through console input whether or not to load saved weights.
    Enter 'y' to load, or 'n' to re-train from scratch.
    Entering anything else raises a ValueError.

    Returns the HandDetectorNet object.
    """
    response = input('Load learned weights?: ').strip().upper()
    if response == 'Y':
        return HandDetectorNet(restore=True, outfile=weight_file)
    elif response == 'N':
        return train_net()
    else:
        raise ValueError('invalid response "{0}"'.format(response))


def prompt_for_save(net: HandDetectorNet) -> None:
    """
    Prompts user through console input whether or not to save the learned weights.
    Enter 'y' to save, or anything else to discard.
    If save, `net` is updated.
    """
    response = input('Save learned weights?: ').strip().upper()
    if response == 'Y':
        net.save()


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


if __name__ == '__main__':
    track_background_subtract(predict_every=5)