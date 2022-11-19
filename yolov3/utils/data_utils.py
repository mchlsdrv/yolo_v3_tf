import logging
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import cv2
from utils.image_funcs import (
    load_image,
)

from utils.logging_funcs import (
    info_log
)

from configs.general_configs import (
    PREPROCESS_IMAGE,
    ORIGINAL_IMAGE_WIDTH,
    ORIGINAL_IMAGE_HEIGHT,
    INFO_BAR_HEIGHT
)


def switch_x_y(bboxes):
    x = bboxes[..., 0].copy()
    y = bboxes[..., 1].copy()

    bboxes[..., 0] = y
    bboxes[..., 1] = x

    return bboxes


def get_top_n_preds(probs, bboxes, top_n, iou_threshold, score_threshold):

    top_n_indices = tf.image.non_max_suppression(
        boxes=switch_x_y(bboxes).reshape(-1, 4),
        scores=probs,
        max_output_size=top_n,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    ).numpy()

    # - Get the top n probabilities
    top_n_probs = probs[top_n_indices]

    # - Get the top n bboxes
    top_n_bboxes = switch_x_y(bboxes).reshape(-1, 4)[top_n_indices]

    return top_n_probs, top_n_bboxes


def get_train_val_split(data: list or np.ndarray, validation_proportion: float = .2, logger: logging.Logger = None):
    # - Find the total number of samples
    n_items = len(data)

    all_idxs = np.arange(n_items)

    # - Choose the items for validation
    val_idxs = np.random.choice(all_idxs, int(validation_proportion * n_items), replace=False)

    # - Choose the items for training
    train_idxs = np.setdiff1d(all_idxs, val_idxs)

    # - Convert the data from list into numpy.ndarray object to use the indexing
    np_data = np.array(data, dtype=object)

    # - The items for training are the once which are not included in the validation set
    train_data = np_data[train_idxs]

    # - Pick the items for the validation set
    val_data = np_data[val_idxs]

    info_log(logger=logger, message=f'| Number of train data files : {len(train_data)} | Number of validation data files : {len(val_data)} |')

    return train_data, val_data


def get_images_bboxes_dict(data: pd.DataFrame, data_root_dir: pathlib.Path, image_files: list or np.ndarray):

    images_dict = {}
    bboxes_dict = {}

    for img_fl in tqdm(image_files):
        images_dict[img_fl] = load_image(
            image_file=str(data_root_dir / img_fl),
            mode=cv2.IMREAD_UNCHANGED,
            preprocess=PREPROCESS_IMAGE,
        )
        bboxes_dict[img_fl] = data.loc[data.loc[:, 'image_id'] == img_fl, 'x, y, w, h'.split(', ')].values

    return images_dict, bboxes_dict


def format_bboxes(image_id: str, label_file: pathlib.Path, x_y_names: tuple, axis: int):

    # - Load the label file
    bboxes = pd.read_csv(label_file)

    # - Rename the columns
    bboxes = bboxes.rename(columns=dict(zip(x_y_names, ('x', 'y'))))
    bboxes = bboxes.loc[bboxes.loc[:, 'axis-0'] == axis]

    # - Fix the outliers - coordinates which overflow the image
    bboxes.loc[bboxes.loc[:, 'x'] < 0, 'x'] = 0.
    bboxes.loc[bboxes.loc[:, 'x'] > ORIGINAL_IMAGE_WIDTH, 'x'] = ORIGINAL_IMAGE_WIDTH
    bboxes.loc[bboxes.loc[:, 'y'] < 0, 'y'] = 0.
    bboxes.loc[bboxes.loc[:, 'y'] > ORIGINAL_IMAGE_HEIGHT - INFO_BAR_HEIGHT, 'y'] = ORIGINAL_IMAGE_HEIGHT - INFO_BAR_HEIGHT

    # - Keep only the important columns
    bboxes = bboxes[['index', 'x', 'y']]

    # - Group-by the index of the bounding box, and choose the minimal coordinates for each x and y
    bboxes_gb = bboxes.groupby('index').agg({
        'x': lambda x: min(list(x)) if x.any() else -1,
        'y': lambda y: min(list(y)) if y.any() else -1
    })

    # - Calculate the width (x_max - x_min), and the height (y_max - y_min)
    # -- The BAR_HEIGHT is the number of pixels which we remove from the bottom as they belong to the status bar
    bboxes_gb_dims = bboxes.groupby('index').agg({
        'x': lambda x: max(list(x)) - min(list(x)) if x.any() else -1,
        'y': lambda y: max(list(y)) - min(list(y)) if y.any() else -1
    })
    bboxes_gb_dims = bboxes_gb_dims.loc[(bboxes_gb_dims.loc[:, 'x'] > -1) & (bboxes_gb_dims.loc[:, 'y'] > -1)]
    # - Rename the columns to w for width and h for height
    bboxes_gb_dims.rename(columns={'x': 'w', 'y': 'h'}, inplace=True)

    # - Add the dimensions to the main DataFrame
    bboxes_gb['w'] = bboxes_gb_dims.loc[:, 'w']
    bboxes_gb['h'] = bboxes_gb_dims.loc[:, 'h']

    # - Remove bboxes with non-positive dimensions
    bboxes_gb = bboxes_gb.loc[(bboxes_gb.loc[:, 'w'] > 0) & (bboxes_gb.loc[:, 'h'] > 0)].reset_index(drop=True)

    # - Add the image id
    bboxes_gb['image_id'] = image_id

    # Sort the columns
    bboxes_gb = bboxes_gb[['image_id', 'x', 'y', 'w', 'h']]

    return bboxes_gb


def correct_bboxes(bboxes: pd.DataFrame):
    bboxes.loc[bboxes.loc[:, 'x'] < 0, 'x'] = 0
    bboxes.loc[bboxes.loc[:, 'x'] > ORIGINAL_IMAGE_WIDTH, 'x'] = ORIGINAL_IMAGE_WIDTH

    bboxes.loc[bboxes.loc[:, 'y'] < 0, 'y'] = 0
    bboxes.loc[bboxes.loc[:, 'y'] > ORIGINAL_IMAGE_HEIGHT - INFO_BAR_HEIGHT, 'y'] = ORIGINAL_IMAGE_HEIGHT - INFO_BAR_HEIGHT
