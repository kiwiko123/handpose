import cv2
import numpy as np
import torch
import torch.utils.data as data
import json
import math
import pathlib
from PIL import Image


class BoundedHandsDataset(data.Dataset):
    def __init__(self, root: str, annotations_file: str, batch_size: int, dimensions: (int, int), transform=None):
        with open(annotations_file) as infile:
            self._annotations = json.load(infile)
        path = pathlib.Path(root)
        self._annotations_list = []
        for image_file in path.iterdir():
            if image_file.stem in self._annotations:
                self._annotations_list.append({image_file.stem: self._annotations[image_file.stem]})

        self._root = pathlib.Path(root)
        self._batch_size = batch_size
        self._dimensions = dimensions
        self.transform = transform


    def __len__(self):
        return len(self._annotations_list)


    def __getitem__(self, index: int) -> {str: np.ndarray}:
        extension = 'jpg'
        info = self._annotations_list[index]
        assert len(info) == 1, 'expected {image_name: [coordinates]}'
        image_name, coordinates = info.popitem()
        image_path = '{0}/{1}.{2}'.format(self._root, image_name, extension)
        image = Image.open(image_path)

        # if image is None:
        #     raise ValueError('failed to read image file at "{0}"'.format(image_path))

        ground_truth_bounding_box = self._get_ground_truth_bounding_box(image, image_name)
        signed_regions = self._sign_regions(image, image_name)
        if self.transform:
            image = self.transform(image)

        labels = torch.Tensor([0] * self._batch_size)
        result = {'image': image,
                  # 'image_path': pathlib.Path(image_path),
                  'bounding_box': ground_truth_bounding_box,
                  'annotations': coordinates,
                  'signed_regions': signed_regions,
                  'labels': labels}

        return result


    def _get_ground_truth_bounding_box(self, image: Image, image_name: str) -> (int,):
        """
        Returns the scaled coordinates of `image`'s ground-truth bounding box, in the form:
        (top-left-x, top-left-y, bottom-right-x, bottom-right-y).

        Coordinates are scaled using self._dimensions.
        """
        coordinates = self._annotations[image_name]
        min_x, min_y = math.inf, math.inf
        max_x, max_y = 0, 0
        for x, y in coordinates:
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)

        scale_x, scale_y = image.size
        down_x, down_y = self._dimensions
        scale_x /= down_x
        scale_y /= down_y

        min_x /= scale_x
        min_y /= scale_y
        max_x /= scale_x
        max_y /= scale_y

        return min_x, min_y, max_x, max_y


    def _sign_regions(self, image: Image, image_name: str) -> np.ndarray:
        """
        Returns a 2D array of binary values representing the gridded image.
        The returned matrix will have a value of 1 at coordinates where the ground-truth bounding box are located,
        and 0 in sections they're not.

        For example, grid[i][j] == 1 means that if `image` is divided into a 13x13 grid,
        the ground-truth bounding box is located in grid (i, j).
        Grid cells that do not contain the bounding box are 0.
        """
        width, height = self._dimensions
        n_grid_x, n_grid_y = (13, 13)
        tl_x, tl_y, br_x, br_y = self._get_ground_truth_bounding_box(image, image_name)
        result = np.zeros((13, 13))

        range_x_start = math.floor((tl_x / width) * n_grid_x)
        range_x_end = math.floor((br_x / width) * n_grid_x)
        range_y_start = math.floor((tl_y / height) * n_grid_y)
        range_y_end = math.floor((br_y / height) * n_grid_y)
        result[range_x_start:range_x_end+1, range_y_start:range_y_end+1] = 1

        # for cx in range(13):
        #     grid_x = cx * (416 / 13)
        #     for cy in range(13):
        #         grid_y = cy * (416 / 13)
        #         if (grid_x >= tl_x and grid_y >= tl_y) and (grid_x >= br_x and grid_y <= br_y):
        #             result[cx, cy] = 1

        return result