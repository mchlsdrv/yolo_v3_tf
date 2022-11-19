import os
import datetime
import sys

import pandas as pd
import pathlib
import wandb

import tensorflow as tf
from wandb.keras import WandbCallback

from configs.general_configs import (
    CONFIGS_DIR_PATH,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    EARLY_STOPPING_VERBOSE,
    REDUCE_LR_ON_PLATEAU_MONITOR,
    REDUCE_LR_ON_PLATEAU_MODE,
    REDUCE_LR_ON_PLATEAU_VERBOSE,
    INFERENCE_IMAGE_DIR,
    DELETE_ON_FINISH
)

from utils.aux_funcs import (
    choose_gpu,
    get_arg_parser,
    get_model,
    get_callbacks,
    get_model_configs,
    get_optimizer
)
from utils.cloud_utils import S3Utils

from utils.logging_funcs import (
    get_logger,
    info_log,
)


'''
You can adjust the verbosity of the logs which are being printed by TensorFlow

by changing the value of TF_CPP_MIN_LOG_LEVEL:
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == '__main__':

    # GENERAL
    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current procedure
    current_run_dir = pathlib.Path(args.output_dir) / f'{TS}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR_PATH / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    # - Configure the GPU to run on
    choose_gpu(gpu_id=args.gpu_id, logger=logger)

    # - Choose the procedure name
    if args.train:
        procedure_name = 'train'
    else:
        procedure_name = 'inference'

    # MODEL
    # - Get the image shape which the network receives
    image_shape = (args.image_size, args.image_size)

    # -1- Build the model and optionally load the weights
    # model_configs = read_yaml(data_file=pathlib.Path(args.model_configs), logger=logger)
    model_configs = get_model_configs(model_name=args.model, configs_dir=args.model_configs_dir, logger=logger)
    model_configs['head_configs']['drop_block']['use'] = not args.no_drop_block
    if model_configs['head_configs']['drop_block']['use']:
        if args.drop_block_keep_prob is not None:
            model_configs['head_configs']['drop_block']['keep_prob'] = args.drop_block_keep_prob
        if args.drop_block_block_size is not None:
            model_configs['head_configs']['drop_block']['block_size'] = args.drop_block_block_size

    model_configs['head_configs']['kernel_regularizer'] = {'type': args.kernel_regularizer_type}
    model_configs['head_configs']['kernel_regularizer'] = {'l1': args.kernel_regularizer_l1}
    model_configs['head_configs']['kernel_regularizer'] = {'l2': args.kernel_regularizer_l2}
    model_configs['head_configs']['kernel_regularizer'] = {'factor': args.kernel_regularizer_factor}
    model_configs['head_configs']['kernel_regularizer'] = {'mode': args.kernel_regularizer_mode}

    model_configs['input_shape'] = image_shape + (1,)
    model, weights_loaded = get_model(
        model_name=args.model,
        model_configs=model_configs,
        checkpoint_dir=pathlib.Path(args.checkpoint_dir) if procedure_name == 'inference' or args.train_continue else pathlib.Path(''),
        bbox_priors_generator_file=args.bbox_priors_generator_file,
        output_dir=current_run_dir if procedure_name == 'train' else pathlib.Path(args.checkpoint_dir).parent,
        logger=logger
    )
    assert procedure_name == 'inference' and model.bbox_priors_gen is not None, f'Could\'nt load the \'{args.bbox_priors_generator_file}\'!'
    assert procedure_name == 'inference' and weights_loaded, f'Could\'nt load the weights from \'{args.checkpoint_dir}\'!'

    info_log(logger=logger, message=f'Weights loaded from {args.checkpoint_dir}: {weights_loaded}')

    # - Weights and biases callback configuration
    if args.wandb:
        wandb.init(
            project='yolo_v3_tf',
            entity='deep-innovations',
        )
        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        }

    s3_utils = S3Utils(
        input_bucket_configs=dict(
            name=args.aws_input_bucket_name,
            region=args.aws_input_region,
            sub_folder=args.aws_input_bucket_subdir,
        ),
        output_bucket_configs=dict(
            name=args.aws_output_bucket_name,
            region=args.aws_output_region,
            sub_folder=args.aws_output_bucket_subdir,
        ),
        delimiter='/',
        logger=logger
    )
    if procedure_name == 'train':

        # DATA
        # - Load the train bbox cvs file
        data_df = pd.read_csv(args.train_bbox_csv_file)
        info_log(logger=logger, message=f'Bounding box .CSV data file preview:\n{data_df.head()}')

        # CALLBACKS
        # - Standard callbacks
        callbacks, tb_th = get_callbacks(
            output_dir=current_run_dir,
            logger=logger
        )
        if args.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=EARLY_STOPPING_MONITOR,
                    min_delta=args.early_stopping_min_delta,
                    patience=args.early_stopping_patience,
                    mode=EARLY_STOPPING_MODE,
                    restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
                    verbose=EARLY_STOPPING_VERBOSE,
                )
            )

        if args.reduce_lr_on_plateau:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=REDUCE_LR_ON_PLATEAU_MONITOR,
                    factor=args.reduce_lr_on_plateau_factor,
                    patience=args.reduce_lr_on_plateau_patience,
                    min_delta=args.reduce_lr_on_plateau_min_delta,
                    cooldown=args.reduce_lr_on_plateau_cooldown,
                    min_lr=args.reduce_lr_on_plateau_min_lr,
                    mode=REDUCE_LR_ON_PLATEAU_MODE,
                    verbose=REDUCE_LR_ON_PLATEAU_VERBOSE,
                )
            )

        if tb_th is not None and args.tensorboard:
            tb_th.start()

        if args.wandb:
            callbacks.append(WandbCallback(log_weights=True))

        # - Optimizer
        optimizer_args = dict(
            learning_rate=args.learning_rate,
            rho=args.optimizer_rho,
            beta_1=args.optimizer_beta_1,
            beta_2=args.optimizer_beta_2,
            amsgrad=args.optimizer_amsgrad,
            momentum=args.optimizer_momentum,
            nesterov=args.optimizer_nesterov,
            centered=args.optimizer_centered,
        )
        optimizer = get_optimizer(algorithm=args.optimizer, args=optimizer_args)

        # - Train the model
        print('''
        ====================
        ===== TRAINING =====
        ====================
        ''')
        model.train(
            data=data_df,
            image_dir=args.train_image_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=optimizer,
            score_threshold=args.train_score_threshold,
            callbacks=callbacks,
            wandb_log=args.wandb
        )

        print('''
        =====================
        ===== INFERENCE =====
        =====================
        ''')
        print(f'Running inference on images at \'{args.inference_image_dir}\'...')

        # - Downloads the images from the bucket to the local directory for the inference
        print(f'Downloading images for detection from \'{args.aws_input_bucket_name}/{args.aws_input_bucket_subdir}\' (region: {args.aws_input_region}) bucket to \'{INFERENCE_IMAGE_DIR}\'...')
        s3_utils.download_files(save_dir=INFERENCE_IMAGE_DIR, delete=DELETE_ON_FINISH)
        print(f'Images for detection were successfully downloaded from \'{args.aws_input_bucket_name}/{args.aws_input_bucket_subdir}\' (region: {args.aws_input_region}) bucket to \'{INFERENCE_IMAGE_DIR}\'!')

        model.detect(
            image_dir=args.inference_image_dir,
            score_threshold=args.inference_score_threshold,
            plot=True,
            wandb_log=args.wandb,
            save_dir=current_run_dir / 'detections',
            logger=logger
        )

        # - Uploads the images with detections from the local directory to the bucket
        print(f'Uploading detections from \'{current_run_dir / "detections/images"}\' to \'{args.aws_output_bucket_name}/{args.aws_output_bucket_subdir}\' (region: {args.aws_output_region}) bucket...')
        s3_utils.upload_files(data_dir=current_run_dir / 'detections/images', delete=DELETE_ON_FINISH)
        print(f'Detections were successfully uploaded from \'{current_run_dir / "detections/images"}\' to \'{args.aws_output_bucket_name}/{args.aws_output_bucket_subdir}\' (region: {args.aws_output_region}) bucket!')

        if tb_th is not None and args.tensorboard:
            tb_th.join()
            print(f'joined the tb thread')
        print(f'Trying to finish...')
        sys.exit(0)

    elif procedure_name == 'inference' and weights_loaded:
        print('''
        =====================
        ===== INFERENCE =====
        =====================
        ''')
        print(f'Running inference on images at \'{args.inference_image_dir}\'...')

        os.makedirs(INFERENCE_IMAGE_DIR, exist_ok=True)
        # - Downloads the images from the bucket to the local directory for the inference
        while True:
            print(f'Looking for images in \'{args.aws_input_bucket_name}/{args.aws_input_bucket_subdir}\' (region: {args.aws_input_region}) bucket...')
            if s3_utils.download_files(save_dir=INFERENCE_IMAGE_DIR, delete=DELETE_ON_FINISH):
                print(f'Images for detection were successfully downloaded from \'{args.aws_input_bucket_name}/{args.aws_input_bucket_subdir}\' (region: {args.aws_input_region}) bucket to \'{INFERENCE_IMAGE_DIR}\'!')

                model.detect(
                    image_dir=args.inference_image_dir,
                    score_threshold=args.inference_score_threshold,
                    plot=True,
                    wandb_log=args.wandb,
                    save_dir=current_run_dir / 'detections',
                    logger=logger
                )

                # - Uploads the images with detections from the local directory to the bucket
                print(f'Uploading detections from \'{current_run_dir / "detections/images"}\' to \'{args.aws_output_bucket_name}/{args.aws_output_bucket_subdir}\' (region: {args.aws_output_region}) bucket...')
                s3_utils.upload_files(data_dir=current_run_dir / 'detections/images', delete=DELETE_ON_FINISH)
                print(f'Detections were successfully uploaded from \'{current_run_dir / "detections/images"}\' to \'{args.aws_output_bucket_name}/{args.aws_output_bucket_subdir}\' (region: {args.aws_output_region}) bucket!')