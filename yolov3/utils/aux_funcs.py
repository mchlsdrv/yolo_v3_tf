import os
from functools import partial

import yaml
import logging
import logging.config
import pickle as pkl
import threading
import multiprocessing as mlp
import argparse
import pathlib
import tensorflow as tf
from custom.models import YOLOv3

from configs.general_configs import (
    IMAGE_SIZE,

    MODEL_CONFIGS_DIR_PATH,

    TRAIN_IMAGE_DIR,
    TRAIN_BBOX_CSV_FILE,

    INFERENCE_IMAGE_DIR,

    OUTPUT_DIR,

    EPOCHS,
    BATCH_SIZE,
    VALIDATION_BATCH_SIZE,
    VALIDATION_PROPORTION,
    LEARNING_RATE,

    TENSOR_BOARD,
    TENSOR_BOARD_WRITE_IMAGES,
    TENSOR_BOARD_WRITE_STEPS_PER_SECOND,
    TENSOR_BOARD_UPDATE_FREQ,

    TENSOR_BOARD_LAUNCH,

    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,

    TERMINATE_ON_NAN,

    REDUCE_LR_ON_PLATEAU_FACTOR,
    REDUCE_LR_ON_PLATEAU_PATIENCE,
    REDUCE_LR_ON_PLATEAU_MIN_DELTA,
    REDUCE_LR_ON_PLATEAU_COOLDOWN,
    REDUCE_LR_ON_PLATEAU_MIN_LR,

    MODEL_CHECKPOINT,
    MODEL_CHECKPOINT_FILE_BEST_MODEL_TEMPLATE,
    MODEL_CHECKPOINT_MONITOR,
    MODEL_CHECKPOINT_VERBOSE,
    MODEL_CHECKPOINT_SAVE_BEST_ONLY,
    MODEL_CHECKPOINT_MODE,
    MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY,
    MODEL_CHECKPOINT_SAVE_FREQ,
    TRAIN_SCORE_THRESHOLD,
    INFERENCE_SCORE_THRESHOLD,

    OPTIMIZER,
    OPTIMIZER_BETA_1,
    OPTIMIZER_RHO,
    OPTIMIZER_BETA_2,
    OPTIMIZER_MOMENTUM,

    KERNEL_REGULARIZER_TYPE,
    KERNEL_REGULARIZER_L1,
    KERNEL_REGULARIZER_L2,
    KERNEL_REGULARIZER_FACTOR,
    KERNEL_REGULARIZER_MODE,

    AWS_INPUT_BUCKET_NAME,
    AWS_INPUT_BUCKET_SUBDIR,
    AWS_INPUT_REGION,

    AWS_OUTPUT_BUCKET_NAME,
    AWS_OUTPUT_BUCKET_SUBDIR,
    AWS_OUTPUT_REGION, CHECKPOINT_DIR, BBOX_PRIORS_GEN_FILE
)
from utils.logging_funcs import (
    info_log
)


def read_yaml(data_file: pathlib.Path):
    data = None
    if data_file.is_file():
        with data_file.open(mode='r') as f:
            data = yaml.safe_load(f.read())
    return data


def decode_file(file):
    if isinstance(file, bytes):
        file = file.decode('utf-8')
    return file


def get_callbacks(output_dir: pathlib.Path, logger: logging.Logger = None):
    callbacks = []
    # -------------------
    # Built-in  callbacks
    # -------------------
    tb_prc = None
    if TENSOR_BOARD:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=output_dir,
                write_images=TENSOR_BOARD_WRITE_IMAGES,
                write_steps_per_second=TENSOR_BOARD_WRITE_STEPS_PER_SECOND,
                update_freq=TENSOR_BOARD_UPDATE_FREQ,
            )
        )
        # - Launch the tensorboard in a thread
        if TENSOR_BOARD_LAUNCH:
            info_log(logger=logger, message=f'Launching a Tensor Board thread on logdir: \'{output_dir}\'...')
            tb_prc = mlp.Process(
                target=lambda: os.system(f'tensorboard --logdir={output_dir}'),
            )

    if TERMINATE_ON_NAN:
        callbacks.append(
            tf.keras.callbacks.TerminateOnNaN()
        )

    if MODEL_CHECKPOINT:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output_dir / MODEL_CHECKPOINT_FILE_BEST_MODEL_TEMPLATE,
                monitor=MODEL_CHECKPOINT_MONITOR,
                verbose=MODEL_CHECKPOINT_VERBOSE,
                save_best_only=MODEL_CHECKPOINT_SAVE_BEST_ONLY,
                mode=MODEL_CHECKPOINT_MODE,
                save_weights_only=MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY,
                save_freq=MODEL_CHECKPOINT_SAVE_FREQ,
            )
        )

    return callbacks, tb_prc


def get_runtime(seconds: float):
    hrs = int(seconds // 3600)
    mins = int((seconds - hrs * 3600) // 60)
    sec = seconds - hrs * 3600 - mins * 60

    # - Format the strings
    hrs_str = str(hrs)
    if hrs < 10:
        hrs_str = '0' + hrs_str
    min_str = str(mins)
    if mins < 10:
        min_str = '0' + min_str
    sec_str = f'{sec:.3}'
    if sec < 10:
        sec_str = '0' + sec_str

    return hrs_str + ':' + min_str + ':' + sec_str + '[H:M:S]'


def find_sub_string(string: str, sub_string: str):
    return True if string.find(sub_string) > -1 else False


def get_file_type(file_name: str):
    file_type = None
    if isinstance(file_name, str):
        dot_idx = file_name.find('.')
        if dot_idx > -1:
            file_type = file_name[dot_idx + 1:]
    return file_type


def get_model_configs(model_name: str, configs_dir: pathlib.Path, logger: logging.Logger):
    model_configs = read_yaml(configs_dir / (model_name + '_configs.yml'))
    if model_configs is not None:
        head_configs = read_yaml(configs_dir / (model_configs.get('head') + '_configs.yml'))
        model_configs['head_configs'] = head_configs
        model_configs['head_configs']['model'] = model_configs.get('head')
        info_log(logger=logger, message=f'The model configs for \'{model_name}\' were loaded from \'{configs_dir}\'')
    else:
        info_log(logger=logger, message=f'No model configs were found for model \'{model_name}\'')
    return model_configs


def get_optimizer(algorithm: str, args: dict):
    optimizer = None
    if algorithm == 'adam':
        optimizer = partial(
            tf.keras.optimizers.Adam,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
            amsgrad=args.get('amsgrad'),
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            tf.keras.optimizers.Nadam,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            tf.keras.optimizers.Adamax,
            beta_1=args.get('beta_1'),
            beta_2=args.get('beta_2'),
        )
    elif algorithm == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad
    elif algorithm == 'adadelta':
        optimizer = partial(
            tf.keras.optimizers.Adadelta,
            rho=args.get('rho'),
        )
    elif algorithm == 'sgd':
        optimizer = partial(
            tf.keras.optimizers.SGD,
            momentum=args.get('momentum'),
            nesterov=args.get('nesterov'),
        )
    elif algorithm == 'rms_prop':
        optimizer = partial(
            tf.keras.optimizers.RMSprop,
            rho=args.get('rho'),
            momentum=args.get('momentum'),
            centered=args.get('centered'),
        )
    return optimizer(learning_rate=args.get('learning_rate'))


def get_model(model_name: str, model_configs: dict, output_dir: pathlib.Path, checkpoint_dir: pathlib.Path = None,  bbox_priors_generator_file: pathlib.Path = None, logger: logging.Logger = None):
    weights_loaded = False

    model = None
    if model_name == 'yolov3':
        model = YOLOv3(
            input_shape=model_configs.get('input_shape'),
            n_grid_cells=model_configs.get('n_grid_cells'),
            n_classes=model_configs.get('n_classes'),
            n_anchor_bboxes=model_configs.get('n_anchor_bboxes'),
            head_configs=model_configs.get('head_configs'),
            output_dir=output_dir,
            logger=logger
        )

        os.makedirs(output_dir, exist_ok=True)
        if bbox_priors_generator_file.is_file():
            with bbox_priors_generator_file.open(mode='rb') as pkl_in:
                model.bbox_priors_gen = pkl.load(pkl_in)
            if isinstance(logger, logging.Logger):
                logger.info(f'bbox_prior_gen was loaded successfully!')
        else:
            logger.exception(f'No priors file was found at \'{bbox_priors_generator_file}\'!')

    if checkpoint_dir.is_dir():
        try:
            latest_cpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_cpt is not None:
                model.load_weights(latest_cpt).expect_partial()
                weights_loaded = True
        except Exception as err:
            if isinstance(logger, logging.Logger):
                logger.exception(f'Can\'t load weighs from \'{checkpoint_dir}\' due to error: {err}')
        else:
            if isinstance(logger, logging.Logger):
                if latest_cpt is not None:
                    logger.info(f'Weights from \'{checkpoint_dir}\' were loaded successfully to the \'RibCage\' model!')
                else:
                    logger.info(f'No weights were found to load in \'{checkpoint_dir}\'!')

    if isinstance(logger, logging.Logger):
        logger.info(model.summary())

    return model, weights_loaded


def choose_gpu(gpu_id: int = 0, logger: logging.Logger = None):
    gpus = tf.config.list_physical_devices('GPU')
    print('Available GPUs:')
    for gpu in gpus:
        print(f'\t- {gpu}')
    if gpus:
        try:
            if gpu_id > -1:
                print(f'Setting GPU: {gpus[gpu_id]}')
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                physical_gpus = tf.config.list_physical_devices('GPU')
                print(f'''
                ====================================================
                > Running on: {physical_gpus}
                ====================================================
                ''')
            else:

                print(f'''
                ====================================================
                > Running on all available devices
                ====================================================
                    ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)


def to_pickle(file, name: str, save_dir: str or pathlib.Path):
    os.makedirs(save_dir, exist_ok=True)

    pkl.dump(file, (save_dir / (name + '.pkl')).open(mode='wb'))


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--model', type=str, choices=['yolov3'], help='The model to use')

    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, len(tf.config.list_physical_devices('GPU')))], default=-1 if len(tf.config.list_physical_devices('GPU')) > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')

    parser.add_argument('--train', default=False, action='store_true', help=f'If to perform the train of the current network')
    parser.add_argument('--train_continue', default=False, action='store_true', help=f'If to continue the training from the checkpoint saved at \'{CHECKPOINT_DIR}\'')
    parser.add_argument('--train_image_dir', type=str, default=TRAIN_IMAGE_DIR, help='The path to the directory where the images are stored')
    parser.add_argument('--train_bbox_csv_file', type=str, default=TRAIN_BBOX_CSV_FILE, help='The path to the .CSV file containing the bounding boxes for the images in \'train_image_dir\'')

    parser.add_argument('--inference', default=False, action='store_true', help=f'If to perform the inference with the current network')
    parser.add_argument('--inference_image_dir', type=str, default=INFERENCE_IMAGE_DIR, help=f'Path to the images to infer directory')

    parser.add_argument('--aws_input_bucket_name', type=str, default=AWS_INPUT_BUCKET_NAME, help=f'Path to the bucket where the images are downloaded from')
    parser.add_argument('--aws_input_region', type=str, default=AWS_INPUT_REGION, help=f'The region of the input bucket')
    parser.add_argument('--aws_input_bucket_subdir', type=str, default=AWS_INPUT_BUCKET_SUBDIR, help=f'The subdirectory where the images are located')

    parser.add_argument('--aws_output_bucket_name', type=str, default=AWS_OUTPUT_BUCKET_NAME, help=f'Path to the bucket where the images are pushed')
    parser.add_argument('--aws_output_region', type=str, default=AWS_OUTPUT_REGION, help=f'The region of output client')
    parser.add_argument('--aws_output_bucket_subdir', type=str, default=AWS_OUTPUT_BUCKET_SUBDIR, help=f'The subdirectory where the images are located')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE, help='The size of the images that will be used for network training and inference. If not specified - the image size will be determined by the value in general_configs.py file.')

    parser.add_argument('--model_configs_dir', type=str, default=MODEL_CONFIGS_DIR_PATH, help='The path to the directory where the configuration of the network are stored (in YAML format)')

    # - TRAINING
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--train_score_threshold', type=float, default=TRAIN_SCORE_THRESHOLD, help=f'The score for detections during the train time')
    parser.add_argument('--inference_score_threshold', type=float, default=INFERENCE_SCORE_THRESHOLD, help=f'The score for detections during the inference time')
    parser.add_argument('--validation_proportion', type=float, default=VALIDATION_PROPORTION, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--validation_batch_size', type=int, default=VALIDATION_BATCH_SIZE, help='The number of samples in each validation batch')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help=f'The path to the directory which contains the checkpoints of the model')
    parser.add_argument('--bbox_priors_generator_file', type=str, default=BBOX_PRIORS_GEN_FILE, help=f'The path to the file which contains the pickled priors generator of the model')

    # - OPTIMIZERS
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'nadam', 'adadelta', 'adamax', 'adagrad', 'rms_prop'], default=OPTIMIZER,  help=f'The optimizer to use')

    parser.add_argument('--optimizer_rho', type=float, default=OPTIMIZER_RHO, help=f'The decay rate (Adadelta, RMSprop)')

    parser.add_argument('--optimizer_beta_1', type=float, default=OPTIMIZER_BETA_1, help=f'The exponential decay rate for the 1st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_beta_2', type=float, default=OPTIMIZER_BETA_2, help=f'The exponential decay rate for the 2st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_amsgrad', default=False, action='store_true', help=f'If to use the Amsgrad function (Adam, Nadam, Adamax)')

    parser.add_argument('--optimizer_momentum', type=float, default=OPTIMIZER_MOMENTUM, help=f'The momentum (SGD, RMSprop)')
    parser.add_argument('--optimizer_nesterov', default=False, action='store_true', help=f'If to use the Nesterov momentum (SGD)')
    parser.add_argument('--optimizer_centered', default=False, action='store_true', help=f'If True, gradients are normalized by the estimated variance of the gradient; if False, by the un-centered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. (RMSprop)')

    # - CALLBACKS
    parser.add_argument('--no_drop_block', default=False, action='store_true', help=f'If to use the drop_block in the network')
    parser.add_argument('--drop_block_keep_prob', type=float, help=f'The probability to keep the block')
    parser.add_argument('--drop_block_block_size', type=int, help=f'The size of the block to drop')

    parser.add_argument('--kernel_regularizer_type', type=str, choices=['l1', 'l2', 'l1l2'], default=KERNEL_REGULARIZER_TYPE, help=f'The type of the regularization')
    parser.add_argument('--kernel_regularizer_l1', type=float, default=KERNEL_REGULARIZER_L1, help=f'The strength of the L1 regularization')
    parser.add_argument('--kernel_regularizer_l2', type=float, default=KERNEL_REGULARIZER_L2, help=f'The strength of the L2 regularization')
    parser.add_argument('--kernel_regularizer_factor', type=float, default=KERNEL_REGULARIZER_FACTOR, help=f'The strength of the orthogonal regularization')
    parser.add_argument('--kernel_regularizer_mode', type=str, choices=['rows', 'columns'], default=KERNEL_REGULARIZER_MODE, help=f"The mode ('columns' or 'rows') of the orthogonal regularization")

    parser.add_argument('--early_stopping', default=False, action='store_true', help=f'If to use the early stopping callback')
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, help=f'The number of epochs to wait for improvement')
    parser.add_argument('--early_stopping_min_delta', type=float, default=EARLY_STOPPING_MIN_DELTA, help=f'The minimal value to count as improvement')

    parser.add_argument('--reduce_lr_on_plateau', default=False, action='store_true', help=f'If to use the learning rate reduction on plateau')
    parser.add_argument('--reduce_lr_on_plateau_patience', type=int, default=REDUCE_LR_ON_PLATEAU_PATIENCE, help=f'The number of epochs to wait for improvement')
    parser.add_argument('--reduce_lr_on_plateau_factor', type=float, default=REDUCE_LR_ON_PLATEAU_FACTOR, help=f'The factor to reduce the lr by')
    parser.add_argument('--reduce_lr_on_plateau_min_delta', type=float, default=REDUCE_LR_ON_PLATEAU_MIN_DELTA, help=f'The minimal value to count as improvement')
    parser.add_argument('--reduce_lr_on_plateau_min_lr', type=float, default=REDUCE_LR_ON_PLATEAU_MIN_LR, help=f'The minimal value of lr after which the training should terminate')
    parser.add_argument('--reduce_lr_on_plateau_cooldown', type=int, default=REDUCE_LR_ON_PLATEAU_COOLDOWN, help=f'The number of improved epochs to restart the count')

    parser.add_argument('--wandb', default=False, action='store_true', help=f'If to use the Weights and Biases board')
    parser.add_argument('--tensorboard', default=False, action='store_true', help=f'If to run tensorboard')

    return parser


def launch_tensorboard(logdir):
    tensorboard_th = threading.Thread(
        target=lambda: os.system(f'tensorboard --logdir={logdir}'),
        daemon=True
    )
    tensorboard_th.start()
    return tensorboard_th
