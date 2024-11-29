import base64
import os
import oss2
import sys

from urllib.parse import urljoin, urlparse, urlunparse
from minio import Minio
from minio.error import S3Error


class OssClient:
    def __init__(self, access_key, access_key_secret, 
                           endpoint,
                           bucket,
                           cdn):

        self.access_key_id = access_key
        self.access_key_secret= access_key_secret
        self.endpoint = endpoint
        self.external_endpoint = endpoint
        self.bucket = bucket
        self.cdn = cdn

        self.storage_type = 'minio'
        if self.storage_type == 'minio':
            self._init_minio()
        else:
            raise('Only support minio at the moment')

    def _init_minio(self):
        """初始化 MinIO 客户端"""
        self.client = Minio(self.endpoint, access_key=self.access_key_id,
                            secret_key=self.access_key_secret, secure=False)

        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

        self.result_url_prefix = urljoin(self.external_endpoint, self.bucket)

    def _init_oss(self):
        """初始化阿里云 OSS 客户端"""
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.client = oss2.Bucket(auth, self.endpoint, self.bucket)

        parsed_url = urlparse(self.external_endpoint)
        new_netloc = self.bucket + "." + parsed_url.netloc

        self.result_url_prefix = urlunparse((parsed_url.scheme, new_netloc, parsed_url.path, parsed_url.params,
                                             parsed_url.query, parsed_url.fragment))

    async def upload_local_file(self, key_prefix: str, file_path: str, name: str) -> str:
        if self.storage_type == 'minio':
            return await self._upload_local_file_minio(key_prefix, file_path, name)
        elif self.storage_type == 'oss':
            return await self._upload_local_file_oss(key_prefix, file_path, name)

    async def upload_image_base64(self, key_prefix: str, name: str, base64_image: str) -> str:
        if self.storage_type == 'minio':
            return await self._upload_image_base64_minio(key_prefix, name, base64_image)
        elif self.storage_type == 'oss':
            return await self._upload_image_base64_oss(key_prefix, name, base64_image)

    def upload_local_file_then_remove(self, key_prefix: str, file_path: str, name: str) -> str:
        if self.storage_type == 'minio':
            return self._upload_local_file_then_remove_minio(key_prefix, file_path, name)
        elif self.storage_type == 'oss':
            return self._upload_local_file_then_remove_oss(key_prefix, file_path, name)

    async def _upload_local_file_minio(self, key_prefix: str, file_path: str, name: str) -> str:
        object_name = key_prefix + name
        try:
            self.client.fput_object(self.bucket, object_name, file_path)
        except S3Error as e:
            logger.error(f"upload_local_file_minio err {e}")
            return ""

        return self.result_url_prefix + "/" + object_name

    async def _upload_image_base64_minio(self, key_prefix: str, name: str, base64_image: str) -> str:
        object_name = key_prefix + name
        try:
            decoded_data = base64.b64decode(base64_image)
            self.client.put_object(self.bucket, key_prefix + name, data=decoded_data, length=len(decoded_data))
        except S3Error as e:
            logger.error(f"upload_image_base64_minio err {e}")
            return ""

        return self.result_url_prefix + "/" + object_name

    def _upload_local_file_then_remove_minio(self, key_prefix: str, file_path: str, name: str) -> str:
        object_name = key_prefix + name
        try:
            self.client.fput_object(self.bucket, object_name, file_path)
            os.remove(file_path)
        except S3Error as e:
            logger.error(f"upload_local_file_then_remove_minio err {e}")
            return ""

        return self.result_url_prefix + "/" + object_name

    async def _upload_local_file_oss(self, key_prefix: str, file_path: str, name: str) -> str:
        object_name = key_prefix + name
        try:
            with open(file_path, 'rb') as file:
                self.client.put_object(object_name, file)
        except Exception as e:
            logger.error(f"upload_local_file_oss err {e}")
            return ""

        return self.result_url_prefix + "/" + object_name

    async def _upload_image_base64_oss(self, key_prefix: str, name: str, base64_image: str) -> str:
        object_name = key_prefix + name
        try:
            self.client.put_object(object_name, base64.b64decode(base64_image))
        except Exception as e:
            logger.error(f"upload_image_base64_oss err {e}")
            return ""

        return urljoin(self.result_url_prefix, self.bucket + "/" + object_name)

    def _upload_local_file_then_remove_oss(self, key_prefix: str, file_path: str, name: str) -> str:
        object_name = key_prefix + name
        try:
            with open(file_path, 'rb') as file:
                self.client.put_object(object_name, file)
            os.remove(file_path)
        except Exception as e:
            logger.error(f"upload_local_file_then_remove_oss err {e}")
            return ""

        return urljoin(self.result_url_prefix, self.bucket + "/" + object_name)

    def sign_url(self, 
                 object_name, 
                 method = "GET",
                 cdn=False,
                 internal=3600, 
                 slash_safe=True):
        if cdn is True:
            return urljoin(self.cdn, object_name)
        
        return self.client.sign_url(method, object_name, internal, slash_safe=slash_safe)

# object_storage = ObjectStorageManager(config=c)

# async def upload_local_file(key_prefix: str, file_path: str, name: str) -> str:
#     return await object_storage.upload_local_file(key_prefix, file_path, name)

# async def upload_image_base64(key_prefix: str, name: str, base64_image: str) -> str:
#     return await object_storage.upload_image_base64(key_prefix, name, base64_image)

# def upload_local_file_then_remove(key_prefix: str, file_path: str, name: str) -> str:
#     return object_storage.upload_local_file_then_remove(key_prefix, file_path, name)







'''
import os

from urllib.parse import urljoin

import oss2


class OssClient():

    def __init__(self, 
        access_key: str, 
        access_key_secret:str, 
        endpoint: str,
        **kwargs):
        self.id = access_key
        self.secret = access_key_secret
        self.endpoint = endpoint
        self._bucket_name = kwargs.pop("bucket", None)
        self.cdn = kwargs.pop("cdn", None)

        self.auth = oss2.Auth(access_key, access_key_secret)

        if self._bucket_name:
            self.bucket_client = oss2.Bucket(self.auth, self.endpoint, self._bucket_name)

    def bucket(self, bucket: str):
        if bucket:
            return oss2.Bucket(self.auth, self.endpoint, bucket)

        return self.bucket_client

    def put_object(self, object_name: str, content):
        self.bucket_client.put_object(object_name, content)
    
    def put_object_from_file(self, 
                             object_name: str, 
                             file,
                             delete_local: bool=True):
        self.bucket_client.put_object_from_file(object_name, file)
        if delete_local:
            os.remove(file)

    def sign_url(self, 
                 object_name, 
                 method = "GET",
                 cdn=False,
                 internal=3600, 
                 slash_safe=True):
        if cdn is True:
            return urljoin(self.cdn, object_name)
        
        return self.bucket_client.sign_url(method, object_name, internal, slash_safe=slash_safe)
'''
