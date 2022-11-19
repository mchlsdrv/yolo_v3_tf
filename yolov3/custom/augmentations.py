import cv2
import numpy as np
import albumentations as A
import albumentations.augmentations.transforms as tr
from albumentations.augmentations.geometric.resize import Resize
from utils.image_funcs import load_image

def train_image_bbox_augmentations(image_width: int, image_height: int):
    return A.Compose(
        [
            A.OneOf([
                A.Flip(),
                A.RandomRotate90(),
            ], p=.5),
            Resize(
                width=image_width,
                height=image_height,
                interpolation=cv2.INTER_CUBIC,
                p=1.
            ),
        ],
        bbox_params={'format': 'coco', 'label_fields': ['labels']}
    )


def train_image_augmentations():
    return A.Compose(
        [
            A.OneOf([
                tr.GaussianBlur(),
                tr.MultiplicativeNoise(),
            ], p=.5),
            A.CLAHE(
                clip_limit=4.,
                tile_grid_size=(8, 8),
                p=1.
            ),
            tr.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=.5
            ),
            A.ToGray(p=1.),
            A.ToFloat(p=1.),
        ],
    )


def validation_augmentations(image_width: int, image_height: int):
    return A.Compose(
        [
            Resize(
                width=image_width,
                height=image_height,
                interpolation=cv2.INTER_CUBIC,
                p=1.
            ),
            A.CLAHE(
                clip_limit=4.,
                tile_grid_size=(8, 8),
                p=1.
            ),
            A.ToGray(p=1.),
            A.ToFloat(p=1.),
        ],
        bbox_params={'format': 'coco', 'label_fields': ['labels']}
    )


def inference_augmentations(image_width: int, image_height: int):
    return A.Compose(
        [
            Resize(
                width=image_width,
                height=image_height,
                interpolation=cv2.INTER_CUBIC,
                p=1.
            ),
            A.CLAHE(
                clip_limit=4.,
                tile_grid_size=(8, 8),
                p=1.
            ),
            A.ToGray(p=1.),
            A.ToFloat(p=1.),
        ],
    )


if __name__ == '__main__':
    img = load_image(image_file='sample.jpg', mode=cv2.IMREAD_UNCHANGED, preprocess=True)
    inf_aug = inference_augmentations(image_width=254, image_height=254)
    aug_img = inf_aug(image=img)['image']
    aug_img = np.expand_dims(cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY), axis=-1)
    print(aug_img.shape)
