import cv2
import numpy as np
import torch
from detector import train_net
import preprocess


def capture_background_image(capturer: cv2.VideoCapture, dimensions=(64, 64), skip=15) -> np.ndarray:
    """
    Captures and returns a single, still image from the webcam to use as the background.
    Use this image as the initial training point for a cv2.BackgroundSubtractorMOG2 object.
    `skip` is the number of frames to skip before capturing the image, to allow for pre-focusing.
    """
    for i in range(skip):
        _ = capturer.read()
    ret, frame = capturer.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=dimensions)
    return image


def track_background_subtract(predict_every=10):
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
    net = train_net()
    capturer = cv2.VideoCapture(0)
    model = cv2.createBackgroundSubtractorMOG2(history=2, detectShadows=False)
    background = capture_background_image(capturer)
    model.apply(background)

    while capturer.isOpened():
        ret, frame = capturer.read()
        image = frame
        if i >= predict_every and i % predict_every == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, dsize=dimensions)
            image = model.apply(image)
            image = np.stack((image,) * 3, -1)
            image = np.reshape(image, (3, 64, 64))
            image = torch.Tensor(image)
            image.unsqueeze_(0)
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



if __name__ == '__main__':
    track_background_subtract(predict_every=5)