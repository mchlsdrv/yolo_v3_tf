import os
import pathlib
import logging
import boto3
from utils.logging_funcs import (
    info_log,
    err_log
)
from botocore.exceptions import EndpointConnectionError, ClientError


class S3Utils:
    def __init__(self, input_bucket_configs: dict, output_bucket_configs: dict, delimiter: str = '/', logger: logging.Logger = None):
        # - INPUT
        self._input_bucket_name = input_bucket_configs.get('name')
        self._input_region = input_bucket_configs.get('region')
        self._input_bucket_sub_folder = input_bucket_configs.get('sub_folder')

        self._input_client = self.set_client(
            region_name=self.input_region
        )
        self._input_bucket = self.set_bucket(
            client=self.input_client,
            bucket_name=self.input_bucket_name,
            region_name=self.input_region
        )

        # - OUTPUT
        self._output_bucket_name = output_bucket_configs.get('name')
        self._output_region = output_bucket_configs.get('region')
        self._output_bucket_sub_folder = output_bucket_configs.get('sub_folder')

        self._output_client = self.set_client(
            region_name=self.output_region
        )
        self._output_bucket = self.set_bucket(
            client=self.output_client,
            bucket_name=self.output_bucket_name,
            region_name=self.output_region
        )

        self._delimiter = delimiter
        self.logger = logger

    def set_client(self, region_name: str):
        # 1) Get the s3 client
        client = None
        try:
            client = boto3.client(
                's3',
                region_name=region_name
            )
        except (ClientError, EndpointConnectionError) as err:
            if self.logger is not None:
                self.logger.exception(err)
        return client

    def set_bucket(self, client, bucket_name: str,  region_name: str):
        bucket = None
        try:
            # - Check if the client is not None
            if client is not None:
                # > Check if the required bucket is already created
                existing_buckets = [bucket['Name'] for bucket in client.list_buckets()['Buckets']]
                if bucket_name not in existing_buckets:
                    # If there is no bucket with the specified name - create a new one
                    if not self.create_new_bucket(client=client, bucket_name=bucket_name, region_name=region_name):
                        # If we could not create a new bucket - return False
                        return False
                bucket = boto3.resource(
                    's3',
                    region_name=region_name
                ).Bucket(bucket_name)
        except (ClientError, EndpointConnectionError) as err:
            if self.logger is not None:
                self.logger.exception(err)
        return bucket

    def get_file_names(self):
        file_names = []
        for fl in self.input_bucket.objects.filter(Delimiter=self.delimiter, Prefix=self.input_bucket_sub_folder):
            file_names.append(fl.key)
        return file_names

    def create_new_bucket(self, client, bucket_name: str, region_name: str) -> bool:
        bucket_created_successfully = False
        if client is not None:
            try:
                if region_name is None:
                    client.create_bucket(Bucket=bucket_name)
                else:
                    location = dict(LocationConstraint=region_name)
                    client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
                bucket_created_successfully = True
            except ClientError as err:
                if self.logger is not None:
                    self.logger.exception(err)
            if self.logger is not None:
                self.logger.info(f'A new bucket named {bucket_name} at {region_name} region was successfully created!')
        else:
            if self.logger is not None:
                self.logger.info(f'Could not create the bucket named {bucket_name} at {region_name} region, because the s3 client is not defined!')
        return bucket_created_successfully

    def _get_file_name(self, file_name):
        strt_idx = -1
        try:
            strt_idx = file_name[::-1].index(self.delimiter)
        except IndexError:
            pass

        if strt_idx > -1:
            file_name = file_name[::-1][:strt_idx][::-1]
        return file_name

    def upload_file(self, file_name: pathlib.Path, key: str, delete: bool = False, logger: logging.Logger = None) -> None:
        try:
            if self.output_bucket is not None and file_name.is_file():
                self.output_bucket.upload_file(
                    Filename=str(file_name),
                    Key=key
                )
                info_log(logger=logger, message=f'The \'{file_name}\' file was successfully uploaded to {key}!')
        except Exception as err:
            err_log(logger=logger, message=f'Could not upload \'{file_name}\' file to {key} due to {err}!')

        if delete:
            try:
                os.remove(file_name)
                info_log(logger=logger, message=f'The \'{file_name}\' file was successfully deleted!')
            except Exception as err:
                err_log(logger=logger, message=f'Could not delete \'{file_name}\' file due to {err}!')

    def upload_files(self, data_dir: pathlib.Path, delete: bool = False, logger: logging.Logger = None) -> bool:
        fls_upld = False
        if data_dir.is_dir():
            files = os.listdir(data_dir)
            if files:
                for fl_nm in files:
                    self.upload_file(
                        file_name=data_dir / str(fl_nm),
                        key=self.output_bucket_sub_folder + fl_nm,
                        delete=delete,
                        logger=logger
                    )
                fls_upld = True
        return fls_upld

    def download_file(self, key: str, file_name: str, save_dir: pathlib.Path, delete: bool = False, logger: logging.Logger = None) -> None:
        try:
            if self.input_bucket is not None:
                os.makedirs(save_dir, exist_ok=True)
                self.input_bucket.download_file(
                    Key=key,
                    Filename=str(save_dir / file_name)
                )
            info_log(logger=logger, message=f'The \'{file_name}\' was successfully downloaded from \'{key}\' and saved at \'{save_dir}\'!')
        except Exception as err:
            err_log(logger=logger, message=f'Could not download \'{file_name}\' file from {key} due to {err}!')

        if delete:
            try:
                self.input_client.delete_object(Bucket=self.input_bucket_name, Key=key)
            except Exception as err:
                err_log(logger=logger, message=f'Could not delete \'{file_name}\' file due to {err}!')

    def download_files(self, save_dir: pathlib.Path, delete: bool = False) -> bool:
        fls_dwnld = False
        file_names = self.get_file_names()
        if file_names[1:]:
            for fl_k in file_names[1:]:
                fl_nm = self._get_file_name(file_name=fl_k)
                self.download_file(
                    key=fl_k,
                    file_name=fl_nm,
                    delete=delete,
                    save_dir=save_dir
                )
            fls_dwnld = True
        return fls_dwnld

    # Properties
    @property
    def input_client(self):
        return self._input_client

    @property
    def input_bucket(self):
        return self._input_bucket

    @property
    def input_bucket_name(self):
        return self._input_bucket_name

    @input_bucket_name.setter
    def input_bucket_name(self, bucket_name: str):
        self._input_bucket_name = bucket_name

    @property
    def input_region(self):
        return self._input_region

    @input_region.setter
    def input_region(self, region):
        self._input_region = region

    @property
    def output_client(self):
        return self._output_client

    @property
    def output_bucket(self):
        return self._output_bucket

    @property
    def output_bucket_name(self):
        return self._output_bucket_name

    @output_bucket_name.setter
    def output_bucket_name(self, bucket_name: str):
        self._output_bucket_name = bucket_name

    @property
    def output_region(self):
        return self._output_region

    @output_region.setter
    def output_region(self, region):
        self._output_region = region

    @property
    def input_bucket_sub_folder(self):
        return self._input_bucket_sub_folder

    @input_bucket_sub_folder.setter
    def input_bucket_sub_folder(self, sub_folder):
        self._input_bucket_sub_folder = sub_folder

    @property
    def output_bucket_sub_folder(self):
        return self._output_bucket_sub_folder

    @output_bucket_sub_folder.setter
    def output_bucket_sub_folder(self, sub_folder):
        self._output_bucket_sub_folder = sub_folder

    @property
    def delimiter(self):
        return self._delimiter

    @delimiter.setter
    def delimiter(self, delimiter):
        self._delimiter = delimiter


if __name__ == '__main__':
    s3_utils = S3Utils(
        input_bucket_configs=dict(
            name='viro-scout-service',
            region='us-east-1',
            sub_folder='input/',
        ),
        output_bucket_configs=dict(
            name='viro-scout-service',
            region='us-east-1',
            sub_folder='output/',
        ),
        delimiter='/',
    )
    # s3_utils.upload_files(data_dir=pathlib.Path('../../../Data/Yeast/batch_2/inference/images'), delete=True)
    s3_utils.download_files(save_dir=pathlib.Path('../../../Data/Yeast/batch_2/inference/images/test'), delete=True)
