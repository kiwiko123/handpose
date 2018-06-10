# Module to preprocess images.
# Running this module as-is assumes that images are located under `./data/test/hand`, and
# preprocessed images will be written under `./data/preprocessed/hand`.
import cv2
import json
import numpy as np
import math
import pathlib
from PIL import Image


# TODO: normalization currently breaks background subtraction
def normalize_image(image: np.ndarray) -> np.ndarray:
    image -= np.mean(image)
    image /= np.std(image)
    return image


def background_separate_image(image: np.ndarray, grayscale=True, inverter=cv2.THRESH_BINARY) -> np.ndarray:
    """
    Source: http://layer0.authentise.com/how-to-track-objects-with-stationary-background.html
    """
    if grayscale:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            message = '{0}\n  load `image` path again using forward slashes'.format(e.args[0])
            e.args = (message,) + e.args[1:]
            raise

    kernel = np.ones((5, 5), np.uint8)
    close_operated_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    _, thresholded = cv2.threshold(close_operated_image, 0, 255, inverter + cv2.THRESH_OTSU)

    return cv2.medianBlur(thresholded, 5)


def preprocess_grayscale(root: pathlib.Path, outdir: pathlib.Path, dimensions=(64, 64)) -> None:
    for image_file in root.iterdir():
        image = cv2.imread(str(image_file), 0)
        if image is not None:
            image = cv2.resize(image, dsize=dimensions)
            outfile = pathlib.Path('{0}/{1}'.format(outdir, image_file.name))
            cv2.imwrite(str(outfile), image)


def create_background_subtractor(**kwargs) -> cv2.BackgroundSubtractorMOG2:
    """
    Utility function that returns a BackgroundSubtractorMOG2 object.
    Calling this function instead of initializing directly ensures the same hyperparameters are used.
    If `kwargs` are provided, initializes a BackgroundSubtractorMOG2 object using those.
    Otherwise, pre-defined parameters are used.
    """
    if not kwargs:
        kwargs = {'history': 1, 'detectShadows': False, 'varThreshold': 50}
    return cv2.createBackgroundSubtractorMOG2(**kwargs)


def preprocess_background_subtraction(background_image: np.ndarray, root: pathlib.Path, outdir: pathlib.Path, dimensions=(64, 64)) -> None:
    """
    Preprocesses images using background subtraction.
    `background_image` is the initial training point so that the `cv2.BackgroundSubtractorMOG2` model knows what to remove.
    Resizes each image to `dimensions`.
    Iterates through every file under `root/`, applies background subtraction, and writes it (same name) under `outdir/`.
    """
    model = create_background_subtractor()
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


def crop_hw3_images(indir: str, outdir: str, padding=0, limit=0) -> None:
    """
    Crops untouched images from the Assignment 3 dataset to show only hands.
    Location of only the hand is determined through the minimum and maximum x/y coordinates of the joint labellings.
    For each image, left (_L) and right (_R) images are saved to `outdir` (each untouched image has 2 hands in it) -
    i.e., 2 * `limit` images will be generated.
    Expects `indir` to be a directory containing:
      - annotation.json
      - Color/ (a subdirectory containing all the untouched images)

    `outdir` is the directory to save all cropped images.
    `padding` is the number of pixels appended to each images' border, in case annotation data is too narrow.
    `limit` is the number of untouched images to process. If <= 0, all images will be processed.
    """
    root = pathlib.Path(indir)
    assert root.is_dir(), 'expected directory "{0}" to exist, with "annotation.json" and "Color/" within.'.format(indir)

    annotations = {}
    with open('{0}/annotation.json'.format(root)) as infile:
        annotations = json.load(infile)

    extension = 'jpg'
    i = 0
    for name, data in annotations.items():
        if limit > 0 and i >= limit:
            break

        min_x, min_y = math.inf, math.inf
        max_x, max_y = 0, 0
        for x, y in data:
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)

        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        full_size_image_name = name[:-2]    # ignore '_L', '_R'
        # normalize (relative) location strings by initializing as Path objects
        image_path = pathlib.Path('{0}/Color/{1}.{2}'.format(root, full_size_image_name, extension))
        write_path = pathlib.Path('{0}/{1}.{2}'.format(outdir, name, extension))

        image = Image.open(str(image_path))
        cropped = image.crop((min_x, min_y, max_x, max_y))
        cropped.save(str(write_path))
        i += 1


if __name__ == '__main__':
    pass
    ###
    ### Crop Assignment 3 images. No need to run this more than once!
    ###
    # crop_hw3_images('./data/Dataset', './data/preprocessed_hw3/hand', padding=5, limit=500)


    ##
    ## Apply background subtraction to images under `./data/custom`.
    ## Run this just once too!
    ##
    # background_image_path = pathlib.Path('data/custom/background.JPG')
    # assert background_image_path.is_file(), 'expected `background.JPG` to be located under `./data/custom`'
    #
    # background_image = cv2.imread(str(background_image_path), 0)
    # training_images = pathlib.Path('./data/custom/hand')
    # preprocessed_images = pathlib.Path('./data/preprocessed/hand')
    # preprocess_background_subtraction(background_image, training_images, preprocessed_images)

    # root = pathlib.Path('./data/preprocessed_hw3/hand')
    # for image_file in root.iterdir():
    #     background = Image.open(str(image_file))
    #     background = background.convert('L')
    #     width, height = background.size
    #     background = background.crop((width * .9, height * .1, width, height))
    #     background = np.asarray(background)
    #     background = cv2.resize(background, dsize=(64, 64))
    #
    #     model = create_background_subtractor()
    #     model.apply(background)
    #
    #     image = cv2.imread(str(image_file), 0)
    #     image = cv2.resize(image, dsize=(64, 64))
    #     mask = model.apply(image)
    #     cv2.imshow('frame', mask)
    #     cv2.waitKey(0)