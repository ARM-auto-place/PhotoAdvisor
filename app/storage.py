from io import BytesIO

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


class StorageError(Exception):
    pass


class S3Storage:
    def __init__(self, settings, client=None):
        self.settings = settings
        kwargs = {}
        if settings.s3_access_key_id:
            kwargs.update(aws_access_key_id=settings.s3_access_key_id.get_secret_value(),
                          aws_secret_access_key=settings.s3_secret_access_key.get_secret_value())
        self.client = client or boto3.client(
            's3', endpoint_url=settings.s3_endpoint_url, region_name=settings.s3_region,
            config=Config(connect_timeout=settings.s3_connect_timeout,
                          read_timeout=settings.s3_read_timeout,
                          retries={'mode': 'standard', 'total_max_attempts': 3},
                          s3={'addressing_style': settings.s3_addressing_style},
                          max_pool_connections=settings.download_workers), **kwargs)

    def validate_key(self, listing_id, key):
        prefix = self.settings.s3_allowed_prefix
        if self.settings.s3_listing_scope:
            prefix += listing_id + '/'
        if not key.startswith(prefix):
            raise StorageError('object_key_not_allowed')

    def fetch(self, listing_id, photo):
        self.validate_key(listing_id, photo.object_key)
        stream = None
        try:
            reference = {}
            if photo.version_id:
                reference['VersionId'] = photo.version_id
            if photo.etag:
                reference['IfMatch'] = photo.etag
            response = self.client.get_object(
                Bucket=self.settings.s3_bucket, Key=photo.object_key, **reference)
            stream = response['Body']
            limit = self.settings.max_image_bytes
            if response.get('ContentLength', 0) > limit:
                raise StorageError('image_too_large')
            # Read in bounded chunks, including when a server omits or misreports length.
            output = BytesIO()
            while True:
                chunk = stream.read(min(65536, limit + 1 - output.tell()))
                if not chunk:
                    break
                output.write(chunk)
                if output.tell() > limit:
                    raise StorageError('image_too_large')
            return output.getvalue()
        except ClientError as exc:
            code = str(exc.response.get('Error', {}).get('Code', ''))
            if code in {'PreconditionFailed', '412', 'NoSuchVersion', 'NoSuchKey', '404',
                        'AccessDenied', '403', 'InvalidObjectState'}:
                raise StorageError('s3_object_unavailable') from exc
            raise StorageError('s3_unavailable') from exc
        except (BotoCoreError, OSError) as exc:
            raise StorageError('s3_unavailable') from exc
        finally:
            if stream is not None:
                stream.close()

    def close(self):
        self.client.close()
