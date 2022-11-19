import pathlib

import numpy as np
import cv2

from configs.general_configs import (
    DEBUG_LEVEL,
)

from utils.preprocessings import (
    preprocess_image
)


def fix_negatives(bboxes: np.ndarray) -> np.ndarray:
    # Fix outliers
    # - Smaller than 0.
    low_outliers = np.argwhere(bboxes < 0)
    low_outliers_x, low_outliers_y = low_outliers[:, 0], low_outliers[:, 1]
    bboxes[low_outliers_x, low_outliers_y] = 0.
    return bboxes


def fix_greater_than_one(bboxes: np.ndarray) -> None:
    # - Grater than 1.
    high_outliers = np.argwhere(bboxes > 1.)
    high_outliers_x, high_outliers_y = high_outliers[:, 0], high_outliers[:, 1]
    bboxes[high_outliers_x, high_outliers_y] = 1.


# def get_images_bboxes_dicts(data_df: pd.DataFrame, data_root_dir: pathlib.Path, image_ids: list, data_type: str, resize_shape: tuple):
#     def _load_image(img_root_dir, img_id):
#         return np.asarray(Image.open(str(img_root_dir / (img_id+'.jpg'))).resize(resize_shape))
#
#     images_dict = {}
#     bboxes_dict = {}
#
#     for img_id in tqdm(image_ids):
#         images_dict[img_id] = _load_image(img_root_dir=data_root_dir / data_type, img_id=img_id)
#         bboxes_dict[img_id] = data_df[img_id].copy() // 4
#
#     return images_dict, bboxes_dict
#
#
# def get_contours(image: np.ndarray):
#
#     # - Find the contours
#     contours, hierarchies = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     # - Find the centroids of the contours
#     centroids = []
#     for contour in contours:
#         M = cv2.moments(contour)
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#             centroids.append((cx, cy))
#     return contours, centroids


def add_channels_dim(image: np.ndarray):
    img = image
    # - If the image is 2D - add the channel dimension
    if len(img.shape) == 2:
        img = np.expand_dims(image, 2)
    return img


def load_image(image_file: str or pathlib.Path, mode: int, preprocess: bool):
    if DEBUG_LEVEL > 2:
        print(f'Loading \'{image_file}\'')

    img = cv2.imread(str(image_file), mode)
    if preprocess:
        img = preprocess_image(image=img)

    return img
