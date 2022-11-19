import cv2
import numpy as np
from configs.general_configs import (
    EPSILON,
    INFO_BAR_HEIGHT,
)


def float_representation(image):
    # - Turn image from 0-255 to 0.-1. float representation
    return image / image.max()


def normalize(image):
    # - Standardization the image
    return (image - image.min()) / (image.max() + EPSILON)


def standardize(image):
    # - Standardization the image
    return (image - image.mean()) / (image.std() + EPSILON)


def to_grayscale(image):
    return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=-1)


def preprocess_image(image: np.ndarray):
    img = image[:-INFO_BAR_HEIGHT, :]
    if len(img.shape) == 3 and img.shape[-1] > 1:
        img = to_grayscale(image=image)
    return img
