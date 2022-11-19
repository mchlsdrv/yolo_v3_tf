import os
import wandb
import pickle as pkl
import cv2
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.utils import tf_utils
from tensorflow.python.ops.numpy_ops import np_config

from utils.plotting_funcs import (
    plot_detections,
    save_figure
)

import logging
from tensorflow.keras import (
    Input,
    Model
)
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Add,
    GlobalAveragePooling2D,
    MaxPool2D,
    Dropout,
    Dense,
    Activation,
    Flatten,
)
from tqdm import tqdm

from configs.general_configs import (
    SHUFFLE,
    VALIDATION_PROPORTION,
    OPTIMIZER,
    METRICS,
    ORIGINAL_IMAGE_WIDTH,
    ORIGINAL_IMAGE_HEIGHT,
    N_TOP_PREDS,
    IOU_THRESHOLD,
    PRIORS_N_CLUSTERS,
    PRIORS_N_ITERATIONS,
    PRIORS_MIN_IMPROVEMENT_DELTA,
    INFO_BAR_HEIGHT,
    PREPROCESS_IMAGE,
    FALSE_POSITIVES_WEIGHT,
    FALSE_NEGATIVES_WEIGHT,
    N_LOGS)
from custom.layers import (
    DropBlock
)
from utils.data_utils import (
    get_top_n_preds,
    get_train_val_split,
    get_images_bboxes_dict,
)

from custom.augmentations import (
    train_image_bbox_augmentations,
    train_image_augmentations,
    validation_augmentations,
    inference_augmentations
)

from utils.image_funcs import (
    load_image,
    fix_negatives,
)

from keras.losses import (
    BinaryCrossentropy,
    MeanSquaredError,
)
from utils.logging_funcs import (
    info_log
)
from custom.callbacks import log_bboxes


plt.style.use('ggplot')
np_config.enable_numpy_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KMeans:
    def __init__(self):
        self.kmeans = None
        self.centroids = None
        self.points = None
        self.markers = {
            0: ['x', 'r'],
            1: ['^', 'g'],
            2: ['*', 'b'],
            3: ['o', 'cyan'],
            4: ['s', 'pink']
        }

    def _preprocess_input(self):
        return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(self.points, dtype=tf.float32), num_epochs=1)

    def fit(self, points: tuple, n_clusters: int, n_iterations: int, min_improvement_delta: float = 0.1):
        self.points = points
        self.kmeans = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=n_clusters,
            use_mini_batch=False
        )

        previous_centroids = self.centroids
        for _ in range(n_iterations):

            # - Expectation: produce the predictions
            self.kmeans.train(self._preprocess_input)

            # - Maximization: adjust the centroids to reduce the error
            self.centroids = self.kmeans.cluster_centers()

            # - Calculate mean absolute improvement, and if it's smaller than the minimal allowed improvement - stop the training
            if previous_centroids is not None:
                delta = np.abs((previous_centroids - self.centroids).mean())
                print(f'K-Means Delta: {delta} < {min_improvement_delta} ? ({delta < min_improvement_delta})')
                if delta < min_improvement_delta:
                    print('Stopping fitting k-means priors... ')
                    break

            # - Update the previous centroids
            previous_centroids = self.centroids

    def predict_clusters(self, points):
        self.points = points
        # print(f'{self.points}, {type(self.points)}')
        return np.array(list(self.kmeans.predict_cluster_index(self._preprocess_input)))

    def get_centroids(self, points):
        clusters = self.predict_clusters(points=points)
        return self.centroids[clusters]

    def plot_centroids(self, points, figsize: tuple = (20, 10), save_dir: pathlib.Path = None):
        fig = ax = None
        if len(self.centroids) <= len(self.markers):
            fig, ax = plt.subplots(figsize=figsize)

            # - Plot the points
            point_clusters = self.predict_clusters(points=points)
            for pnt, clstr in zip(points, point_clusters):
                mrkr, clr = self.markers.get(clstr)
                ax.scatter(pnt[0], pnt[1], marker=mrkr, c=clr)

            # - Plot the centroids
            for clstr, cntr in enumerate(self.centroids):
                ax.scatter(cntr[0], cntr[1], c='k', marker=self.markers.get(clstr)[0])

            if isinstance(save_dir, pathlib.Path):
                save_figure(figure=fig, save_file=save_dir / 'centroids.png')
        return fig, ax


class VGG16(tf.keras.Model):
    def __init__(self, input_shape: tuple, output_size: int, model_configs: dict):
        super().__init__()

        self.input_layer = Input(shape=input_shape)
        self.conv2d_1 = Conv2D(**model_configs.get('conv2d_1'))
        self.conv2d_2 = Conv2D(**model_configs.get('conv2d_2'))
        self.conv2d_3 = Conv2D(**model_configs.get('conv2d_3'))
        self.conv2d_4 = Conv2D(**model_configs.get('conv2d_4'))
        self.max_pool2d = MaxPool2D(**model_configs.get('max_pool2d'))
        self.dense_layer = Dense(**model_configs.get('dense_layer'))
        self.output_layer = Dense(units=output_size, activation='softmax')

    def call(self, inputs, training: bool = False, **kwargs):

        # - Input
        X = self.input_layer(inputs)

        # - Block I
        X = self.conv2d_1(X)
        X = MaxPool2D(X)

        # - Block II
        X = self.conv2d_2(X)
        X = self.conv2d_2(X)
        X = self.max_pool2d(X)

        # - Block III
        X = self.conv2d_3(X)
        X = self.conv2d_3(X)
        X = self.conv2d_3(X)
        X = self.max_pool2d(X)

        # - Block IV
        X = self.conv2d_4(X)
        X = self.conv2d_4(X)
        X = self.conv2d_4(X)
        X = self.max_pool2d(X)
        X = self.conv2d_4(X)
        X = self.conv2d_4(X)
        X = self.conv2d_4(X)
        X = self.max_pool2d(X)

        X = Flatten(X)

        X = self.dense_layer(X)

        X = self.dense_layer(X)

        X = self.output_layer(X)

        return X


class ResNet(tf.keras.Model):
    class ResBlock(tf.keras.layers.Layer):
        def __init__(self, filters: tuple, kernel_sizes: tuple, strides: tuple = ((1, 1), (1, 1)), activations: tuple = ('relu', 'relu'), paddings: tuple = ('same', 'same'), dilation_rates: tuple = ((1, 1), (1, 1))):
            super().__init__()

            # I) - First conv block
            self.conv2d_1 = Conv2D(
                filters[0], kernel_sizes[0], strides=strides[0], activation=activations[0], padding=paddings[0], dilation_rate=dilation_rates[0])
            self.batch_norm_1 = BatchNormalization()

            # II) - Second conv block
            self.conv2d_2 = Conv2D(
                filters[1], kernel_sizes[1], strides=strides[1], activation=None, padding=paddings[1], dilation_rate=dilation_rates[1])
            self.batch_norm_2 = BatchNormalization()

            # III) - Skip connection
            self.identity = Conv2D(filters[1], 1, padding='same')
            self.shortcut = Add()

            # IV) - Activation
            self.activation = Activation(activations[1])

        def call(self, inputs, training=False):
            x = self.conv2d_1(inputs)
            x = self.batch_norm_1(x)
            x = self.conv2d_2(x)
            x = self.batch_norm_2(x)

            if x.shape[1:] == inputs.shape[1:]:
                x = self.shortcut([x, inputs])
            else:
                x = self.shortcut([x, self.identity(inputs)])

            return self.activation(x)

    def __init__(self, input_shape: tuple, output_size: int, net_configs: dict):
        super().__init__()

        self.input_image_shape = input_shape
        # 1) Input layer
        self.input_layer = tf.keras.Input(shape=self.input_image_shape)

        self.conv2d_1 = Conv2D(**net_configs.get('conv2d_1'))

        self.conv2d_2 = Conv2D(**net_configs.get('conv2d_2'))

        self.max_pool2d = MaxPool2D(**net_configs.get('max_pool_2d'))

        # 2) ResBlocks
        res_blocks_configs = net_configs.get('res_blocks')

        conv2_block_configs = res_blocks_configs.get('conv2_block_configs')
        self.conv2_blocks = []
        for idx in range(conv2_block_configs.get('n_blocks')):
            self.conv2_blocks.append(self.ResBlock(**conv2_block_configs.get('block_configs')))

        conv3_block_configs = res_blocks_configs.get('conv3_block_configs')
        self.conv3_blocks = []
        for idx in range(conv3_block_configs.get('n_blocks')):
            self.conv3_blocks.append(self.ResBlock(**conv3_block_configs.get('block_configs')))

        conv4_block_configs = res_blocks_configs.get('conv4_block_configs')
        self.conv4_blocks = []
        for idx in range(conv4_block_configs.get('n_blocks')):
            self.conv4_blocks.append(self.ResBlock(**conv4_block_configs.get('block_configs')))

        conv5_block_configs = res_blocks_configs.get('conv5_block_configs')
        self.conv5_blocks = []
        for idx in range(conv5_block_configs.get('n_blocks')):
            self.conv5_blocks.append(self.ResBlock(**conv5_block_configs.get('block_configs')))

        self.conv2d_3 = Conv2D(**net_configs.get('conv2d_3'))

        self.global_avg_pool = GlobalAveragePooling2D()

        self.dense_layer = Dense(**net_configs.get('dense_layer'))

        self.dropout_layer = Dropout(**net_configs.get('dropout_layer'))

        self.output_layer = Dense(units=output_size, **net_configs.get('output_layer'))

    def call(self, inputs, training=False):
        X = self.conv2d_1(inputs)

        X = self.conv2d_2(X)

        X = self.max_pool2d(X)

        for conv2_block in self.conv2_blocks:
            X = conv2_block(X)

        for conv3_block in self.conv3_blocks:
            X = conv3_block(X)

        for conv4_block in self.conv4_blocks:
            X = conv4_block(X)

        for conv5_block in self.conv5_blocks:
            X = conv5_block(X)

        X = self.conv2d_3(X)

        X = self.global_avg_pool(X)

        X = self.dense_layer(X)

        X = self.dropout_layer(X)

        X = self.output_layer(X)

        return X

    def model(self):
        x = tf.keras.Input(shape=self.input_image_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class DarkNet53(tf.keras.Model):
    def __init__(self, input_shape: tuple, output_size: int, model_configs: dict, logger: logging.Logger = None):
        super().__init__()

        self.model = None
        self.input_image_shape = input_shape
        self.output_size = output_size
        self.model_configs = model_configs
        self.kernel_regularizer = self._get_kernel_regularizer(configs=model_configs.get('kernel_regularizer'))
        self.logger = logger
        self._build_net()

    @staticmethod
    def _get_kernel_regularizer(configs: dict):
        kernel_regularizer = None
        if configs.get('type') == 'l1':
            kernel_regularizer = tf.keras.regularizers.L1(l1=configs.get('l1'))
        elif configs.get('type') == 'l2':
            kernel_regularizer = tf.keras.regularizers.L2(l2=configs.get('l2'))
        elif configs.get('type') == 'l1l2':
            kernel_regularizer = tf.keras.regularizers.L2(l1=configs.get('l1'), l2=configs.get('l2'))
        elif configs.get('type') == 'orthogonal':
            kernel_regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=configs.get('factor'), l2=configs.get('mode'))
        return kernel_regularizer

    def _build_net(self):
        # == INPUT ==
        print(f'\nWorking on:\n\t>Input layers')
        X_input = Input(shape=self.input_image_shape)

        X = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X_input)
        X = BatchNormalization()(X)

        # == BLOCK 1 ==
        print(f'\nWorking on:\n\t>Block 1')

        X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        X_sc = X

        for _ in tqdm(range(2)):
            X = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Add()([X_sc, X])
            if self.model_configs.get('drop_block')['use']:
                X = DropBlock(keep_prob=self.model_configs.get('drop_block')['keep_prob'], block_size=self.model_configs.get('drop_block')['block_size'])(X)

            X_sc = X

        # == BLOCK 2 ==
        print(f'\nWorking on:\n\t>Block 2')

        X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        X_sc = X

        for _ in tqdm(range(2)):
            X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = tf.keras.layers.Add()([X_sc, X])
            if self.model_configs.get('drop_block')['use']:
                X = DropBlock(keep_prob=self.model_configs.get('drop_block')['keep_prob'], block_size=self.model_configs.get('drop_block')['block_size'])(X)

            X_sc = X

        # == BLOCK 3 ==
        print(f'\nWorking on:\n\t>Block 3')
        X = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        X_sc = X

        for _ in tqdm(range(8)):
            X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Add()([X_sc, X])
            if self.model_configs.get('drop_block')['use']:
                X = DropBlock(keep_prob=self.model_configs.get('drop_block')['keep_prob'], block_size=self.model_configs.get('drop_block')['block_size'])(X)

            X_sc = X

        # == BLOCK 4 ==
        print(f'\nWorking on:\n\t>Block 4')
        X = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        X_sc = X

        for _ in tqdm(range(8)):
            X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Add()([X_sc, X])
            if self.model_configs.get('drop_block')['use']:
                X = DropBlock(keep_prob=self.model_configs.get('drop_block')['keep_prob'], block_size=self.model_configs.get('drop_block')['block_size'])(X)

            X_sc = X

        # == BLOCK 5 ==
        print(f'\nWorking on:\n\t>Block 5')

        X = Conv2D(1024, (3, 3), strides=(2, 2), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        X_sc = X

        for _ in tqdm(range(4)):
            X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
            X = BatchNormalization()(X)

            X = Add()([X_sc, X])
            if self.model_configs.get('drop_block')['use']:
                X = DropBlock(keep_prob=self.model_configs.get('drop_block')['keep_prob'], block_size=self.model_configs.get('drop_block')['block_size'])(X)

            X_sc = X

        # == OUTPUT ==
        print(f'\nWorking on:\n\t>Output layers')

        X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='swish', kernel_regularizer=self.kernel_regularizer)(X)
        X = BatchNormalization()(X)

        # TODO: How to make the output of the grid shape ?
        outputs = Conv2D(self.output_size, (1, 1), strides=(1, 1), activation='relu', kernel_regularizer=self.kernel_regularizer)(X)

        self.model = Model(inputs=X_input, outputs=outputs)

        print(f'\n===\nModel was build successfully!\n===\n')
        print(self.model.summary())


class YOLOv3(tf.keras.Model):
    """
    Information on the model may be found here:
        => https://www.semanticscholar.org/paper/YOLOv3%3A-An-Incremental-Improvement-Redmon-Farhadi/e4845fb1e624965d4f036d7fd32e8dcdd2408148
    """
    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, image_ids: list, images_dict: dict, image_width: int, image_height: int, n_grid_cells: int, n_anchor_bboxes: int, output_size: int, image_bbox_augmentations, image_augmentations, labels_dict: dict = None, batch_size=1, shuffle=True):
            self.image_ids = image_ids

            self.images_dict = images_dict
            self.n_grid_cells = n_grid_cells
            self.n_anchor_bboxes = n_anchor_bboxes
            self.output_shape = (self.n_grid_cells, self.n_grid_cells, output_size)
            self.labels_dict = labels_dict
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indexes = np.arange(len(self.image_ids))
            if self.shuffle:
                np.random.shuffle(self.indexes)
            self.img_bbox_augs = image_bbox_augmentations
            self.img_augs = image_augmentations

            self.image_grid = self.get_image_grid(n_grid_cells, image_width, image_height)
            self.grid_cell_width = image_width / self.n_grid_cells
            self.grid_cell_height = image_height / self.n_grid_cells

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.image_ids))

            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __len__(self):
            return int(np.floor(len(self.image_ids) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

            batch_image_ids = [self.image_ids[i] for i in indexes]

            return self.__get_batch(batch_image_ids)

        def __get_batch(self, batch_image_ids):
            X, y = [], []

            for idx, image_id in enumerate(batch_image_ids):
                batch_image = self.images_dict[image_id]
                batch_bboxes = self.labels_dict[image_id]

                # Fix negatives
                batch_bboxes = fix_negatives(bboxes=batch_bboxes)

                batch_image, batch_bboxes = self.augment_image(batch_image, batch_bboxes)

                X.append(batch_image)
                y.append(batch_bboxes)

            X = tf.convert_to_tensor(np.array(X), dtype=tf.float32)
            y = tf.convert_to_tensor(np.array(y), dtype=tf.float32)
            return X, y

        def augment_image(self, image, bboxes):

            # - Augmentations which are applied to the image and the bboxes together (e.g., resize, rotate, flip etc.)
            if self.img_bbox_augs is not None:
                bbox_labels = np.ones(len(bboxes))
                img_bbox_aug_result = self.img_bbox_augs(image=image, bboxes=bboxes, labels=bbox_labels)
                image = np.array(img_bbox_aug_result['image'])
                bboxes = np.array(img_bbox_aug_result['bboxes'])

            # - Augmentations which are applied to the image only
            if self.img_augs is not None:
                image = self.img_augs(image=image)['image']

            label_grid = self.get_label_grid(bboxes)

            return image, label_grid

        def get_label_grid(self, bboxes):
            label_grid = np.zeros(self.output_shape)

            for row in range(self.n_grid_cells):
                for column in range(self.n_grid_cells):
                    cell_coordinates = self.image_grid[row, column]

                    lbl = self.get_label(bboxes=bboxes, cell_coordinates=cell_coordinates)

                    # - Add the anchor box to the grid
                    label_grid[row, column] = lbl

            return label_grid

        def get_label(self, bboxes, cell_coordinates):
            cell_x, cell_y, cell_width, cell_height = cell_coordinates
            cell_x_max = cell_x + cell_width
            cell_y_max = cell_y + cell_height

            # - Create an empty set of anchor bboxes, each with 5 entries for (c, x, y, w, h)
            ancr_bxs = np.zeros((self.n_anchor_bboxes, 5))
            free_ancr_bx_idx = 0
            # - Try adding the bboxes to the anchor bboxes.
            # (*) Not all the bboxes will be added, in case there are more bboxes than anchor bboxes
            # for idx in range(self.n_anchor_bboxes):
            for bbox in bboxes:
                bbox_x, bbox_y, bbox_width, bbox_height = bbox

                # -- Compute the center coordinates of the bbox
                bbox_center_x = bbox_x + (bbox_width / 2)
                bbox_center_y = bbox_y + (bbox_height / 2)

                # -- If the bbox center falls inside the current cell grid
                if cell_x <= bbox_center_x < cell_x_max and cell_y <= bbox_center_y < cell_y_max:
                    # - Once we add a bbox to anchor box we break, because we don't want to be changing it in case another bbox fits,
                    # but to add it to another anchor box, if there is one free left
                    if free_ancr_bx_idx >= self.n_anchor_bboxes:
                        break

                    # - Put certainty of 1. because it's a ground truth bbox
                    ancr_bxs[free_ancr_bx_idx][0] = 1.

                    # - Put the yolo represented bbox
                    ancr_bxs[free_ancr_bx_idx][1:] = self.coco_2_yolo(
                        coco_bbox=(bbox_x, bbox_y, bbox_width, bbox_height),
                        cell_shape=(cell_x, cell_y, cell_width, cell_height),
                    )

                    free_ancr_bx_idx += 1

            # - Join the anchor boxes together to produce the label
            lbl = np.concatenate(ancr_bxs, axis=0)

            return lbl

        @staticmethod
        def get_image_grid(n_grid_cells, image_width, image_height):
            image_grid = np.zeros((n_grid_cells, n_grid_cells, 4))

            # initial cell coordinates
            cell = [0, 0, int(image_width / n_grid_cells), int(image_height / n_grid_cells)]

            for row in range(n_grid_cells):
                for column in range(n_grid_cells):
                    image_grid[row, column] = cell
                    cell[0] = cell[0] + cell[2]
                cell[0] = 0
                cell[1] = cell[1] + cell[3]

            return image_grid

        @staticmethod
        def coco_2_yolo(coco_bbox, cell_shape):
            bbox_x, bbox_y, bbox_width, bbox_height = coco_bbox
            cell_x, cell_y, cell_width, cell_height = cell_shape

            # Move the top left x, y coordinates to the center, and make them relative to the cell
            # by the following relations:
            # 1) b_x = (x + b_w / 2 - c_x) / c_w
            # 2) b_y = (y + b_h / 2 - c_y) / c_h
            bbox_relative_center_x = (bbox_x + (bbox_width / 2) - cell_x) / cell_width
            bbox_relative_center_y = (bbox_y + (bbox_height / 2) - cell_y) / cell_height

            # Change the bbox width and height relative to the cell width and height
            # by the following relations:
            # 1) b_w = b_w / c_w
            # 2) b_h = b_h / c_h
            bbox_relative_w = bbox_width / cell_width
            bbox_relative_h = bbox_height / cell_height

            return bbox_relative_center_x, bbox_relative_center_y, bbox_relative_w, bbox_relative_h

        @staticmethod
        def yolo_2_coco(yolo_bboxes, image_width, image_height, n_grid_cells):
            image_grid = YOLOv3.DataGenerator.get_image_grid(
                n_grid_cells=n_grid_cells,
                image_width=image_width,
                image_height=image_height
            )

            # - Reshape the bboxes to sit on top of each other to modify all the coordinates together
            original_shape = yolo_bboxes.shape
            coco_bboxes = np.expand_dims(yolo_bboxes, axis=-2)  # yolo_bboxes.reshape((*original_shape[:-1], -1, 5))

            # - Extract the bboxes
            bboxes = coco_bboxes[..., 1:5]

            # - Descale width,height
            # by the following relations:
            # 1) b_w = b_w * c_w
            # 2) b_h = b_h * c_h
            bboxes[..., 2:4] = bboxes[..., 2:4] * image_grid[..., np.newaxis, 2:4]

            # - Descale x, y and move them from the center to the top-left corner
            # by the following relations:
            # 1) x' = b_x * c_w
            # 2) y' = b_y * c_h
            bboxes[..., :2] = bboxes[..., :2] * image_grid[..., np.newaxis, 2:]

            # 3) x' = x' + c_x
            # 4) y' = y' + c_y
            bboxes[..., :2] = bboxes[..., :2] + image_grid[..., np.newaxis, :2]

            # 5) x' = x' - b_w / 2
            # 6) y' = y' - b_h / 2
            bboxes[..., :2] = bboxes[..., :2] - bboxes[..., 2:] / 2

            bboxes[..., :2] = fix_negatives(bboxes=bboxes[..., :2])
            # - Update the bboxes in the yolo preds
            coco_bboxes[..., 1:5] = bboxes

            # - Reshape the bboxes to their original shape
            coco_bboxes = coco_bboxes.reshape(original_shape)

            return coco_bboxes

    class YoloOutLayer(tf.keras.layers.Layer):
        """
        The confidence and the x, y coordinates of the center are activated with the sigmoid function, while the width and the max_height
        of the bbox are activated with the exponent
        """

        def __init__(self):
            super().__init__()

        def call(self, inputs, **kwargs):
            orig_shp = inputs.shape
            inputs = tf.reshape(inputs, (*orig_shp[:-1], -1, 5))
            c_x_y, w_h = tf.math.sigmoid(inputs[..., :3]), tf.math.log1p(inputs[..., 3:5])
            preds = tf.reshape(tf.concat([c_x_y, w_h], axis=-1), orig_shp)
            return preds

        @tf_utils.shape_type_conversion
        def compute_output_shape(self, input_shape):
            return input_shape

    def __init__(self, input_shape: tuple, n_grid_cells: int, n_classes: int, n_anchor_bboxes: int, head_configs: dict, output_dir: pathlib.Path, logger: logging.Logger = None):
        super().__init__()

        self.head = None
        self.input_image_shape = input_shape
        self.n_grid_cells = n_grid_cells
        self.n_classes = n_classes
        self.n_anchor_bboxes = n_anchor_bboxes
        self.output_size = self.n_anchor_bboxes * (5 + self.n_classes) if self.n_classes > 1 else self.n_anchor_bboxes * 5
        self.head_configs = head_configs
        self.output_dir = output_dir
        self.logger = logger
        self.optimizer = OPTIMIZER

        self.bce_loss = BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

        self.mse_loss = MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

        self.score_threshold = None

        self.epoch = 0

        self.log_interval = None

        self.wandb_log = False

        self.grid_cell_width = None
        self.grid_cell_height = None
        self.bbox_priors_gen = None

        self.output_layer = self.YoloOutLayer()

        self._get_head()

    def _get_head(self):
        if self.head_configs.get('model') == 'darknet53':
            self.head = DarkNet53(input_shape=self.input_image_shape, output_size=self.output_size, model_configs=self.head_configs, logger=self.logger).model

    def _get_data_generator(self, data: pd.DataFrame, image_dir: pathlib.Path, image_ids: list, batch_size: int, image_bbox_augmentations, image_augmentations):
        # - Get the dictionaries with the image and its corresponding bounding boxes
        images_dict, bboxes_dict = get_images_bboxes_dict(
            data=data,
            data_root_dir=image_dir,
            image_files=image_ids,
        )
        # - Get the data loader
        data_generator = self.DataGenerator(
            image_ids=image_ids,
            images_dict=images_dict,
            image_width=self.input_image_shape[0],
            image_height=self.input_image_shape[1],
            labels_dict=bboxes_dict,
            n_grid_cells=self.n_grid_cells,
            n_anchor_bboxes=self.n_anchor_bboxes,
            output_size=self.output_size,
            batch_size=batch_size,
            shuffle=SHUFFLE,
            image_bbox_augmentations=image_bbox_augmentations,
            image_augmentations=image_augmentations
        )

        return data_generator

    def _loss_func(self, y_true, y_pred):

        # - Reshape the preds to manipulate the entries
        y_true = y_true.reshape((self.n_grid_cells, self.n_grid_cells, -1, 5))
        y_pred = y_pred.reshape((self.n_grid_cells, self.n_grid_cells, -1, 5))

        # TODO: Class loss
        # c_loss = binary_crossentropy(
        #     y_true[..., 5:],
        #     y_pred[..., 5:],
        # )
        # - Controls the trade off between false negatives to the false positives

        # - As we want the loss only apply to object which have a target in the image we mask all the losses of objects without targets
        obj_mask = y_true[..., 0]

        obj_weights = tf.concat(
            tf.where(
                y_true[..., 0] == 0,
                FALSE_POSITIVES_WEIGHT,
                FALSE_NEGATIVES_WEIGHT,
                ), axis=0
        )
        pred_obj = obj_mask * (y_pred[..., 0] + tf.abs((y_true[..., 0] - y_pred[..., 0])) * obj_weights)
        obj_loss = self.mse_loss(
            y_true[..., 0],
            pred_obj
        )

        pred_bboxes = obj_mask[..., np.newaxis] * y_pred[..., 1:5]
        bbox_loss = self.mse_loss(
            y_true[..., 1:5],
            pred_bboxes
        )

        return obj_loss + bbox_loss

    def _get_bbox_priors_gen(self, bboxes):
        # self.bbox_priors_gen = KMeans(n_clusters=5, random_state=0).fit(bboxes)
        self.bbox_priors_gen = KMeans()
        self.bbox_priors_gen.fit(points=bboxes, n_clusters=PRIORS_N_CLUSTERS, n_iterations=PRIORS_N_ITERATIONS, min_improvement_delta=PRIORS_MIN_IMPROVEMENT_DELTA)
        # TODO: Upload the scatter plot of centroids to wandb
        _, _ = self.bbox_priors_gen.plot_centroids(points=bboxes, save_dir=self.output_dir / 'priors')
        # wandb.log({'Bbox width and height prior clusters': cntrd_fig})
        self._save_bbox_prior_gen(save_dir=self.output_dir / 'priors')

    def _get_bbox_prior_dims(self, bboxes):
        bbox_priors = self.bbox_priors_gen.get_centroids(points=bboxes)
        return bbox_priors

    def _get_priors(self, bboxes):
        bboxes_shape = bboxes.shape

        # - Get the prior bbox width and heights
        bboxes_rshp = bboxes.reshape((*bboxes_shape[:-1], -1, 5))
        bbox_prior_dims = self._get_bbox_prior_dims(bboxes=bboxes_rshp[..., 3:].reshape((-1, 2))).reshape((*bboxes_shape[:-1], 2))

        priors = np.zeros_like(bboxes)
        priors[..., 3:5] = bbox_prior_dims

        return priors

    def call(self, inputs, training: bool = False):

        # - Run the conv net
        outputs = self.head(inputs)

        # - Apply the YOLO activation layer
        preds = self.output_layer(outputs)

        return preds

    def _add_priors(self, bboxes):
        result = bboxes
        if bboxes.any():
            # - Calculate the priors to add to the predictions
            priors = self._get_priors(bboxes=bboxes)

            # - Add the priors to the outputs
            result = bboxes + priors

        return result

    def _get_scale_factor(self, image_width, image_height):
        x_scale_fct, y_scale_fct = image_width / self.input_image_shape[0], image_height / self.input_image_shape[1]
        return x_scale_fct, y_scale_fct

    def __scale_bbox_coords(self, bboxes, image_width, image_height):
        # - Save the bboxes initial shape
        bboxes_shape = bboxes.shape

        # - Reshape the bboxes to manipulate their coordinates
        bboxes = bboxes.reshape((*bboxes_shape[:-1], -1, 5))

        # - Calculate the x and y scale factors
        x_scale_fct, y_scale_fct = self._get_scale_factor(image_width=image_width, image_height=image_height)

        # - Descale bboxes
        bboxes[..., 1] = bboxes[..., 1] / x_scale_fct
        bboxes[..., 2] = bboxes[..., 2] / y_scale_fct

        # - Reshape the bboxes back to their original shape
        bboxes = bboxes.reshape(bboxes_shape)

        return bboxes

    def __scale_bbox_dims(self, bboxes, image_width, image_height):
        # - Save the bboxes initial shape
        bboxes_shape = bboxes.shape

        # - Reshape the bboxes to manipulate their coordinates
        bboxes = bboxes.reshape((*bboxes_shape[:-1], -1, 5))

        # - Calculate the x and y scale factors
        x_scale_fct, y_scale_fct = self._get_scale_factor(image_width=image_width, image_height=image_height)

        # - Descale bboxes
        bboxes[..., 3] = bboxes[..., 3] / x_scale_fct
        bboxes[..., 4] = bboxes[..., 4] / y_scale_fct

        # - Reshape the bboxes back to their original shape
        bboxes = bboxes.reshape(bboxes_shape)

        return bboxes

    def __descale_bbox_coords(self, bboxes, image_width, image_height):
        # - Save the bboxes initial shape
        bboxes_shape = bboxes.shape

        # - Reshape the bboxes to manipulate their coordinates
        bboxes = np.expand_dims(bboxes, axis=-2)

        # - Calculate the x and y scale factors
        x_scale_fct, y_scale_fct = self._get_scale_factor(image_width=image_width, image_height=image_height)

        # - Descale bboxes
        bboxes[..., 1] = bboxes[..., 1] * x_scale_fct
        bboxes[..., 2] = bboxes[..., 2] * y_scale_fct

        # - Reshape the bboxes back to their original shape
        bboxes = bboxes.reshape(bboxes_shape)

        return bboxes

    def __descale_bbox_dims(self, bboxes, image_width, image_height):
        # - Save the bboxes initial shape
        bboxes_shape = bboxes.shape

        # - Reshape the bboxes to manipulate their coordinates
        bboxes = np.expand_dims(bboxes, axis=-2)

        # - Calculate the x and y scale factors
        x_scale_fct, y_scale_fct = self._get_scale_factor(image_width=image_width, image_height=image_height)

        # - Descale bboxes
        bboxes[..., 3] = bboxes[..., 3] * x_scale_fct
        bboxes[..., 4] = bboxes[..., 4] * y_scale_fct

        # - Reshape the bboxes back to their original shape
        bboxes = bboxes.reshape(bboxes_shape)

        return bboxes

    def _descale_bboxes(self, bboxes, image_width, image_height):
        if bboxes.any():
            bboxes = self.__descale_bbox_coords(bboxes=bboxes, image_width=image_width, image_height=image_height)
            bboxes = self.__descale_bbox_dims(bboxes=bboxes, image_width=image_width, image_height=image_height)
        return bboxes

    def save(self, save_path: pathlib.Path, **kwargs):
        self.head.save(save_path)

    def _save_bbox_prior_gen(self, save_dir: pathlib.Path):
        with (save_dir / 'bbox_priors_gen.pkl').open(mode='wb') as pkl_out:
            pkl.dump(self.bbox_priors_gen, pkl_out)
            info_log(logger=self.logger, message=f'bbox_prior_gen was successfully saved to \'{save_dir}/bbox_prior_gen.pkl\'')

    def summary(self):
        return self.head.summary()

    def __log_labels(self, log_type: str, images: np.ndarray, labels: np.ndarray):

        # - Labels
        all_lbls = []
        for img, lbls in zip(images, labels):
            # - Convert from YOLO to COCO format
            bboxes = self.DataGenerator.yolo_2_coco(
                yolo_bboxes=lbls,
                image_width=self.input_image_shape[0],
                image_height=self.input_image_shape[1],
                n_grid_cells=self.n_grid_cells,
            )

            # - Reshape the bboxes to manipulate by coordinate
            bboxes = bboxes.reshape((self.n_grid_cells, self.n_grid_cells, -1, 5))

            # - Convert from (x, y, w, h) -> (x_min, y_min, x_max, y_max)
            bboxes[..., 3:5] = bboxes[..., 3:5] + bboxes[..., 1:3]

            # - Choose only the non-zero labels (i.e., cells which have an object)
            non_zro_lbl_idxs = np.argwhere(lbls.reshape((-1, 5))[..., 0] > 0).flatten()
            bboxes = bboxes.reshape((-1, 5))[non_zro_lbl_idxs]

            all_lbls.append(log_bboxes(img, bboxes=bboxes[..., 1:], scores=bboxes[..., 0], class_labels=['yeast']))

        if self.wandb_log:
            wandb.log({f'{log_type} labels': all_lbls})

    def __log_detections(self, log_type: str, images: np.ndarray, detections: np.ndarray, score_threshold: float, final: bool = False):
        all_dets = []
        for img, dets in zip(images, detections):
            # - Convert from YOLO to COCO format, clear low confidence detections, add priors and optionally descale
            if not final:
                dets = self._get_final_detections(
                    bboxes=dets,
                    score_threshold=score_threshold
                )

            # - Convert from (x, y, w, h) -> (x_min, y_min, x_max, y_max)
            dets[..., 3:5] = dets[..., 3:5] + dets[..., 1:3]

            all_dets.append(log_bboxes(img, bboxes=dets[..., 1:5], scores=dets[..., 0], class_labels=['yeast']))

        if self.wandb_log:
            wandb.log({f'{log_type} detections': all_dets})

    def _log_samples(self, log_type: str, images: np.ndarray, labels: np.ndarray, detections: np.ndarray, score_threshold: float):
        self.__log_labels(log_type=log_type, images=images, labels=labels)

        self.__log_detections(log_type=log_type, images=images, detections=detections, score_threshold=score_threshold)

    def train_step(self, data) -> dict:

        # - Get the data of the current epoch
        imgs, true_bboxes = data

        # - Compute the loss according to the predictions
        with tf.GradientTape() as tape:
            pred_bboxes = self(imgs, training=True)
            loss = self._loss_func(true_bboxes, pred_bboxes)

        # - Get the weights to adjust according to the loss calculated
        trainable_vars = self.trainable_variables

        # - Calculate gradients
        gradients = tape.gradient(loss, trainable_vars)

        # - Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if self.epoch % self.log_interval == 0:
            self._log_samples(
                log_type='train',
                images=imgs.numpy(),
                labels=true_bboxes.numpy(),
                detections=pred_bboxes.numpy(),
                score_threshold=self.score_threshold
            )

        # - Return the mapping metric names to current value
        return {'loss': loss.numpy().mean()}

    def test_step(self, data) -> dict:

        # - Get the data of the current epoch
        imgs, true_bboxes = data

        # - Compute the loss according to the predictions
        pred_bboxes = self(imgs, training=False)
        loss = self._loss_func(true_bboxes, pred_bboxes)
        if self.wandb_log:
            wandb.log({'val_loss': loss})

        if self.epoch % self.log_interval == 0:
            self._log_samples(
                log_type='validation',
                images=imgs.numpy(),
                labels=true_bboxes.numpy(),
                detections=pred_bboxes.numpy(),
                score_threshold=self.score_threshold
            )

        self.epoch += 1

        return {'loss': loss.numpy().mean()}

    def train(self, data: pd.DataFrame, image_dir: pathlib.Path, epochs: int, batch_size: int, optimizer, score_threshold: float, callbacks: list, wandb_log: bool = False):
        # -1- Prepare the data
        # - Get the names of the images present in the data
        image_ids = np.unique(data.loc[:, 'image_id'].values)

        # - Split the data into train and validation
        train_image_ids, val_image_ids = get_train_val_split(
            data=image_ids,
            validation_proportion=VALIDATION_PROPORTION,
        )

        # - Get the train data generator
        train_generator = self._get_data_generator(
            data=data,
            image_dir=image_dir,
            image_ids=train_image_ids,
            batch_size=batch_size,
            image_bbox_augmentations=train_image_bbox_augmentations(image_width=self.input_image_shape[0], image_height=self.input_image_shape[1]),
            image_augmentations=train_image_augmentations()
        )
        self.grid_cell_width = train_generator.grid_cell_width
        self.grid_cell_height = train_generator.grid_cell_height

        # - Get the validations data generator
        val_generator = self._get_data_generator(
            data=data,
            image_dir=image_dir,
            image_ids=val_image_ids,
            batch_size=batch_size,
            image_bbox_augmentations=validation_augmentations(image_width=self.input_image_shape[0], image_height=self.input_image_shape[1]),
            image_augmentations=None
        )

        # - Get the bbox_priors_gen
        x_scale_fct, y_scale_fct = self._get_scale_factor(image_width=ORIGINAL_IMAGE_WIDTH, image_height=ORIGINAL_IMAGE_HEIGHT - INFO_BAR_HEIGHT)
        train_bboxes_width = data.loc[:, 'w'].values / (x_scale_fct * self.grid_cell_width)
        train_bboxes_height = data.loc[:, 'h'].values / (y_scale_fct * self.grid_cell_height)
        train_bboxes_dims = np.vstack([train_bboxes_width, train_bboxes_height]).T

        self._get_bbox_priors_gen(bboxes=train_bboxes_dims)

        # -2- Compile the model
        self.optimizer = optimizer
        self.compile(
            loss=self._loss_func,
            optimizer=self.optimizer,
            run_eagerly=True,
            metrics=METRICS
        )
        # -3- Fit the model
        self.log_interval = max(epochs // N_LOGS, 1)
        self.wandb_log = wandb_log
        self.score_threshold = score_threshold
        history = self.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    @staticmethod
    def _clear_low_conf_bboxes(preds, score_threshold):

        # - Reshape the bboxes to sit on top of each other to modify all the coordinates together
        original_shape = preds.shape

        preds = preds.reshape((original_shape[0], original_shape[1], -1, 5))

        # - Extract the object probabilities
        probs = preds[..., 0]

        # - Extract the bboxes
        bboxes = preds[..., 1:5].astype(np.int16)

        top_n_probs, top_n_bboxes = get_top_n_preds(
            probs=probs.flatten(),
            bboxes=bboxes,
            top_n=N_TOP_PREDS,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=score_threshold
        )

        # - Combine the probs and the bboxes
        top_n_preds = np.zeros((len(top_n_probs), 5))
        top_n_preds[:, 0] = top_n_probs
        top_n_preds[:, 1:5] = top_n_bboxes

        return top_n_preds

    def _get_final_detections(self, bboxes: np.ndarray, score_threshold: float, image_width: np.ndarray = None, image_height: np.ndarray = None):
        """
        This function
        I) Transforms each bounding box from a YOLO representation back to the COCO format, by:
             - Reshaping the bounding box from 0-1 scale back to 0-size of the image (256 in our case).
             - Changing the (x, y) coordinate from the center of the bounding box to the top left corner.
             - Changing the width and height back to x_max, y_max (i.e., VOC shape).
        II) Clears bboxes which are of a low object certainty (configured in configs/general_configs.py by SCORE_THRESHOLD)
        III) Adds the priors to the bboxes
        IV) (Optional) Descales the bboxes to fit custom-sized images
        """
        # - Convert the preds from YOLO to COCO format
        final_preds = self.DataGenerator.yolo_2_coco(
            yolo_bboxes=bboxes,
            image_width=self.input_image_shape[0],
            image_height=self.input_image_shape[1],
            n_grid_cells=self.n_grid_cells,
        )

        # - Clear low confidence bboxes
        final_preds = self._clear_low_conf_bboxes(
            preds=final_preds,
            score_threshold=score_threshold
        )

        # - Add the priors to the bboxes
        final_preds = self._add_priors(
            bboxes=final_preds
        )

        if image_width is not None and image_height is not None:
            final_preds = self._descale_bboxes(
                bboxes=final_preds,
                image_width=image_width,
                image_height=image_height
            )

        return final_preds

    def get_image_detections(self, image: np.ndarray, score_threshold: float, save_name: str, save_dir: pathlib.Path, plot: bool = False) -> tuple:

        inf_augs = inference_augmentations(image_width=self.input_image_shape[0], image_height=self.input_image_shape[1])

        aug_img = inf_augs(image=image)['image']

        # - Add the batch dimension
        aug_img = np.expand_dims(aug_img, axis=0)

        # - Get the predictions
        bboxes = self(aug_img).numpy()

        # - Clear low confidence predictions
        detections = self._get_final_detections(
            bboxes=bboxes,
            score_threshold=score_threshold
        )

        if plot:
            plot_detections(
                images=aug_img,
                bboxes=[detections],
                file_names=[save_name],
                save_dir=save_dir / 'images'
            )

        return aug_img, detections

    def detect(self, image_dir: pathlib.Path, score_threshold: float, save_dir: pathlib.Path, plot: bool, wandb_log: bool = False, logger: logging.Logger = None) -> np.ndarray:

        os.makedirs(save_dir, exist_ok=True)

        img_fls = os.listdir(image_dir)
        print(f'Detecting images:\n{img_fls}\n...')

        img_2_dets = {}
        imgs = []
        dets = []
        for img_fl in tqdm(img_fls):
            img = load_image(
                image_file=str(image_dir / img_fl),
                mode=cv2.IMREAD_UNCHANGED,
                preprocess=PREPROCESS_IMAGE,
            )
            img_nm = img_fl[::-1][img_fl[::-1].index('.') + 1:][::-1]
            print(f'Detecting image \'{img_fl}\'...')
            img, img_dets = self.get_image_detections(
                image=img,
                score_threshold=score_threshold,
                save_name=img_nm,
                save_dir=save_dir,
                plot=plot,
            )

            imgs.append(img)
            dets.append(img_dets)

            # - Add the bboxes to the predictions
            img_2_dets[img_fl] = img_dets

        if wandb_log:
            self.__log_detections(
                log_type='inference',
                images=np.array(imgs),
                detections=np.array(dets),
                score_threshold=score_threshold,
                final=True
            )

        info_log(logger=logger, message=f'Detections:\n\t{img_2_dets}')
        pkl.dump(img_2_dets, (save_dir / 'detections.pkl').open(mode='wb'))

        return np.array(dets)

    def get_config(self):
        pass
