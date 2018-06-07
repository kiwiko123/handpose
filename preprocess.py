# Module to preprocess images.
# Running this module as-is assumes that images are located under `./data/test/hand`, and
# preprocessed images will be written under `./data/preprocessed/hand`.
import cv2
import numpy as np
import pathlib


# TODO: normalization currently breaks background subtraction
def normalize_image(image: np.ndarray) -> np.ndarray:
    image -= np.mean(image)
    image /= np.std(image)
    return image


# def background_separate_image(image: np.ndarray, grayscale=True, inverter=cv2.THRESH_BINARY) -> np.ndarray:
#     """
#     Source: http://layer0.authentise.com/how-to-track-objects-with-stationary-background.html
#     """
#     if grayscale:
#         try:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         except cv2.error as e:
#             message = '{0}\n  load `image` path again using forward slashes'.format(e.args[0])
#             e.args = (message,) + e.args[1:]
#             raise
#
#     kernel = np.ones((5, 5), np.uint8)
#     close_operated_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#     _, thresholded = cv2.threshold(close_operated_image, 0, 255, inverter + cv2.THRESH_OTSU)
#
#     return cv2.medianBlur(thresholded, 5)
#
#
# def preprocess_images(root: pathlib.Path, outdir: pathlib.Path, dimensions=(256, 256)) -> None:
#     for image_file in root.iterdir():
#         image = cv2.imread(str(image_file), 0)
#         if image is not None:
#             image = background_separate_image(image, grayscale=False)
#             resized = cv2.resize(image, dimensions)
#             cv2.imwrite('{0}/{1}'.format(outdir, image_file.name), resized)


def preprocess_background_subtraction(background_image: np.ndarray, root: pathlib.Path, outdir: pathlib.Path, dimensions=(64, 64)) -> None:
    """
    Preprocesses images using background subtraction.
    `background_image` is the initial training point so that the `cv2.BackgroundSubtractorMOG2` model knows what to remove.
    Resizes each image to `dimensions`.
    Iterates through every file under `root/`, applies background subtraction, and writes it (same name) under `outdir/`.
    """
    model = cv2.createBackgroundSubtractorMOG2(history=2, detectShadows=False)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    background_image = cv2.resize(background_image, dsize=dimensions)
    model.apply(background_image)

    for image_file in root.iterdir():
        image = cv2.imread(str(image_file), 0)
        if image is not None:
            image = cv2.resize(image, dsize=dimensions)
            mask = model.apply(image)
            cv2.imwrite('{0}/{1}'.format(outdir, image_file.name), mask)


if __name__ == '__main__':
    background_image_path = pathlib.Path('data/custom/background.JPG')
    assert background_image_path.is_file(), 'expected `background.JPG` to be located under `./data/custom`'

    background_image = cv2.imread('data/custom/background.JPG', 0)
    training_images = pathlib.Path('./data/custom/hand')
    preprocessed_images = pathlib.Path('./data/preprocessed/hand')
    preprocess_background_subtraction(background_image, training_images, preprocessed_images)
