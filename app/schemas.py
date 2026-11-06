from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Photo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    photo_id: str = Field(min_length=1, max_length=128)
    object_key: str = Field(min_length=1, max_length=1024)
    version_id: str | None = Field(default=None, min_length=1, max_length=1024)
    etag: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator('etag')
    @classmethod
    def exact_etag(cls, value):
        if value is None:
            return value
        token = value[1:-1] if value.startswith('"') and value.endswith('"') else value
        if not token or any(c in token for c in '"*,\\') or any(ord(c) <= 32 or ord(c) == 127 for c in token):
            raise ValueError('etag must identify one exact object')
        return value

    @field_validator('object_key')
    @classmethod
    def safe_key(cls, value):
        if (value.startswith('/') or '\\' in value or '%' in value or ':' in value
                or any(ord(c) < 32 or ord(c) == 127 for c in value)
                or any(p in {'', '.', '..'} for p in value.split('/'))):
            raise ValueError('invalid object key')
        return value


class EvaluateParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    listing_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    photos: list[Photo] = Field(max_length=100)
    locale: Literal['ru', 'en', 'hy'] = 'ru'

    @model_validator(mode='after')
    def unique_ids(self):
        if len({p.photo_id for p in self.photos}) != len(self.photos):
            raise ValueError('duplicate photo_id')
        return self


class PhotoResult(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    photo_id: str
    category: str = 'unknown'
    candidate: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_kind: str = 'not_calibrated'
    score: float | None = Field(default=None, ge=0, le=1)
    score_kind: str | None = None
    similarity: float | None = Field(default=None, ge=-1, le=1)
    margin: float | None = Field(default=None, ge=0, le=2)
    recognition: Literal['known', 'uncertain', 'unknown', 'error'] = 'unknown'
    quality: Literal['good', 'poor', 'uncertain', 'unavailable'] = 'unavailable'
    quality_issues: list[str] = Field(default_factory=list)
    accepted: bool = False
    duplicate_of: str | None = None
    duplicate_kind: str | None = None
    similar_to: str | None = None
    error: str | None = None
    detection: dict = Field(default_factory=dict)
    alternatives: list[dict] = Field(default_factory=list)
