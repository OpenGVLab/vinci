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

    
