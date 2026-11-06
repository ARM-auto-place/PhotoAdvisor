from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


POLICY_VERSION = 'photo-advisor-v2'


class Threshold(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    min_similarity: float = Field(ge=-1, le=1)
    min_margin: float = Field(ge=0, le=2)


class Calibration(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    validated: bool = Field(default=False, strict=True)
    dataset_id: str | None = None
    model_version: str | None = None
    taxonomy_hash: str | None = None
    policy_version: Literal['photo-advisor-v2'] | None = None
    unknown_similarity: float = Field(ge=-1, le=1)
    classes: dict[str, Threshold] = Field(default_factory=dict)
    notes: str = ''

    @model_validator(mode='after')
    def provenance(self):
        if self.validated and not all((self.dataset_id, self.model_version,
                                       self.taxonomy_hash, self.policy_version, self.classes)):
            raise ValueError('validated calibration requires thresholds and full provenance')
        return self


def validate_calibration(raw, model_version, taxonomy_hash, labels):
    calibration = Calibration.model_validate(raw)
    if set(calibration.classes) - set(labels):
        raise ValueError('unknown calibration classes')
    if calibration.validated and (calibration.model_version != model_version
                                  or calibration.taxonomy_hash != taxonomy_hash):
        raise ValueError('calibration does not match model/taxonomy')
    return calibration.model_dump()
