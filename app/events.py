from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas import Photo


REQUIRED = (
    'front', 'rear', 'left_side', 'right_side', 'engine_bay', 'trunk',
    'under_trunk', 'interior', 'fender_front_left', 'fender_front_right',
    'fender_rear_left', 'fender_rear_right',
)


class ObjectPhoto(Photo):
    bucket: str = Field(min_length=1, max_length=255)
    manual_category: Literal[*REQUIRED] | None = None

    @model_validator(mode='after')
    def immutable_reference(self):
        if not self.version_id and not self.etag:
            raise ValueError('version_id or etag required')
        if self.version_id == 'null' and not self.etag:
            raise ValueError('mutable null version requires etag')
        return self


class PhotoEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    listing_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    listing_revision: int = Field(ge=1, le=2**63-1, strict=True)
    event_type: Literal['image.upserted', 'image.deleted', 'photoset.finalized']
    image_id: str | None = Field(default=None, min_length=1, max_length=128)
    photo: ObjectPhoto | None = None
    photos: list[ObjectPhoto] | None = Field(default=None, max_length=100)

    @model_validator(mode='after')
    def event_shape(self):
        if self.event_type == 'photoset.finalized':
            if self.photos is None or self.photo is not None or self.image_id is not None:
                raise ValueError('finalized requires a complete photos manifest only')
            if len({p.photo_id for p in self.photos}) != len(self.photos):
                raise ValueError('duplicate image ID')
        elif self.event_type == 'image.upserted':
            if not self.photo or self.image_id != self.photo.photo_id or self.photos is not None:
                raise ValueError('upsert requires matching image_id and photo')
        elif self.image_id is None or self.photo is not None or self.photos is not None:
            raise ValueError('delete requires image_id only')
        return self
