from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    service_token: SecretStr = SecretStr('')
    s3_endpoint_url: str | None = None
    s3_region: str = 'us-east-1'
    s3_bucket: str = ''
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_addressing_style: str = 'path'
    s3_allowed_prefix: str = 'listings/'
    s3_listing_scope: bool = True
    s3_connect_timeout: int = Field(3, ge=1, le=30)
    s3_read_timeout: int = Field(10, ge=1, le=60)
    max_photos: int = Field(20, ge=1, le=100)
    max_image_bytes: int = Field(12 * 1024 * 1024, ge=1024)
    max_pixels: int = Field(24_000_000, ge=10000)
    max_rpc_bytes: int = Field(65536, ge=1024)
    download_workers: int = Field(2, ge=1, le=8)
    batch_size: int = Field(4, ge=1, le=20)
    cpu_threads: int = Field(2, ge=1, le=32)
    device: str = 'cpu'
    cv_backend: Literal['clip', 'trained'] = 'clip'
    trained_model_dir: Path = Path('models/view-classifier-v2')
    model_dir: Path = Path('models/clip')
    detector_path: Path = Path('models/ssdlite.pt')
    checklist_path: Path = Path('configs/checklist.json')
    calibration_path: Path = Path('configs/calibration.json')
    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = ''
    kafka_input_topic: str = ''
    kafka_result_topic: str = ''
    kafka_dlq_topic: str = ''
    kafka_group_id: str = ''
    kafka_security_protocol: str = 'PLAINTEXT'
    kafka_sasl_mechanism: str = 'PLAIN'
    kafka_sasl_username: str | None = None
    kafka_sasl_password: SecretStr | None = None
    kafka_ssl_cafile: str | None = None
    kafka_max_attempts: int = Field(3, ge=1, le=10)
    kafka_retry_backoff: float = Field(1, ge=0, le=30)
    state_path: Path = Path('data/advisor.sqlite3')
    kafka_checklist_path: Path = Path('configs/checklist-kafka.json')

    @field_validator('s3_endpoint_url', 's3_access_key_id', 's3_secret_access_key',
                     'kafka_ssl_cafile', 'kafka_sasl_username', 'kafka_sasl_password', mode='before')
    @classmethod
    def empty_to_none(cls, value):
        return None if value == '' else value

    @model_validator(mode='after')
    def validate_security(self):
        if self.cv_backend == 'trained' and self.checklist_path == Path('configs/checklist.json'):
            self.checklist_path = self.kafka_checklist_path
        if self.kafka_enabled:
            if not all((self.kafka_bootstrap_servers, self.kafka_input_topic,
                        self.kafka_result_topic, self.kafka_dlq_topic, self.kafka_group_id)):
                raise ValueError('configure Kafka servers, topics and group before enabling Kafka')
            if len({self.kafka_input_topic, self.kafka_result_topic, self.kafka_dlq_topic}) != 3:
                raise ValueError('Kafka input/result/DLQ topics must be distinct')
            if self.kafka_security_protocol not in {'PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL'}:
                raise ValueError('invalid Kafka security protocol')
            if self.kafka_security_protocol.startswith('SASL') and not (
                    self.kafka_sasl_username and self.kafka_sasl_password):
                raise ValueError('configure Kafka SASL credentials')
        if self.s3_addressing_style not in {'path', 'virtual', 'auto'}:
            raise ValueError('invalid S3 addressing style')
        if not self.s3_allowed_prefix or not self.s3_allowed_prefix.endswith('/'):
            raise ValueError('S3_ALLOWED_PREFIX must be a nonempty folder prefix')
        if bool(self.s3_access_key_id) != bool(self.s3_secret_access_key):
            raise ValueError('provide both S3 credential fields or use IAM')
        return self
