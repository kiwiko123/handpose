# Module to preprocess images.
# Running this module as-is assumes that images are located under `./data/test/hand`, and
# preprocessed images will be written under `./data/preprocessed/hand`.
import cv2
import numpy as np
import pathlib


def background_separate_image(image: np.ndarray, grayscale=False, inverter=cv2.THRESH_BINARY) -> np.ndarray:
    """
    Source: http://layer0.authentise.com/how-to-track-objects-with-stationary-background.html
    """
    if grayscale:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            message = '{0}\n  load `image` path again using forward slashes'.format(e.args[0])
            e.args = (message,) + e.args[1:]
            raise

    kernel = np.ones((5, 5), np.uint8)
    close_operated_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    _, thresholded = cv2.threshold(close_operated_image, 0, 255, inverter + cv2.THRESH_OTSU)

    return cv2.medianBlur(thresholded, 5)


def preprocess_images(root: pathlib.Path, outdir: pathlib.Path) -> None:
    for image_file in root.iterdir():
        image = cv2.imread(str(image_file), 0)
        if image is not None:
            image = background_separate_image(image, grayscale=False)
            cv2.imwrite('{0}/{1}'.format(outdir, image_file.name), image)


def track_webcam():
    capturer = cv2.VideoCapture(0)

    while capturer.isOpened():
        ret, frame = capturer.read()
        if ret:
            image = background_separate_image(frame, grayscale=True)
            cv2.imshow('frame', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capturer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    training_images = pathlib.Path('data/test/hand')
    preprocessed_images = pathlib.Path('data/preprocessed/hand')
    preprocess_images(training_images, preprocessed_images)