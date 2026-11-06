import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
import numpy as np

from app.imaging import decode, duplicate_kind, fingerprint, quality
from app.policy import POLICY_VERSION
from app.rules import summarize
from app.schemas import PhotoResult
from app.storage import StorageError

logger = logging.getLogger(__name__)


class Advisor:
    def __init__(self, settings, storage, vision):
        self.settings, self.storage, self.vision = settings, storage, vision
        self.pool = ThreadPoolExecutor(max_workers=settings.download_workers)
        self.lock = Lock()

    def prepare(self, listing_id, photo):
        image = None
        try:
            data = self.storage.fetch(listing_id, photo)
            image, size = decode(data, self.settings.max_pixels)
            return image, quality(image, size), fingerprint(data, image)
        except (StorageError, ValueError) as exc:
            if image is not None:
                image.close()
            return str(exc)
        except (OSError, cv2.error):
            if image is not None:
                image.close()
            return 'image_processing_failed'

    def evaluate(self, params):
        results, seen = [], []
        # One request owns the model at a time. API rejects concurrent work with retryable busy error.
        for offset in range(0, len(params.photos), self.settings.batch_size):
            photos = params.photos[offset:offset+self.settings.batch_size]
            prepared = list(self.pool.map(lambda p: self.prepare(params.listing_id, p), photos))
            usable = [(p, item) for p, item in zip(photos, prepared) if not isinstance(item, str)]
            try:
                try:
                    predictions = self.vision.predict([item[0] for _, item in usable]) if usable else []
                    if len(predictions) != len(usable):
                        raise ValueError('inference output count mismatch')
                    vector_size = seen[0][2].shape if seen else None
                    for prediction in predictions:
                        vector = np.asarray(prediction['embedding'])
                        if (vector.ndim != 1 or not vector.size or not np.isfinite(vector).all()
                                or (vector_size is not None and vector.shape != vector_size)):
                            raise ValueError('invalid embedding')
                        vector_size = vector.shape
                        PhotoResult(photo_id='validation', **{
                            k: v for k, v in prediction.items() if k != 'embedding'})
                except Exception:
                    logger.error('inference_batch_failed')
                    predictions = [None] * len(usable)
                predictions = iter(predictions)
                for photo, item in zip(photos, prepared):
                    if isinstance(item, str):
                        results.append(PhotoResult(photo_id=photo.photo_id, recognition='error', error=item))
                        continue
                    image, issues, fp = item
                    prediction = next(predictions)
                    if prediction is None:
                        results.append(PhotoResult(photo_id=photo.photo_id, recognition='error', error='inference_failed'))
                        continue
                    prediction = prediction.copy()
                    vector = np.asarray(prediction.pop('embedding'))
                    prediction['quality_issues'] = issues + prediction['quality_issues']
                    result = PhotoResult(photo_id=photo.photo_id, **prediction)
                    hard_issues = {'too_small', 'blurred', 'underexposed', 'overexposed'}
                    result.quality = ('poor' if hard_issues.intersection(result.quality_issues)
                                      else 'uncertain' if result.quality_issues else 'good')
                    result.accepted = result.recognition == 'known' and result.quality == 'good'
                    for prior, prior_fp, prior_vector in seen:
                        # Compare with surviving representatives only. Otherwise a rejected
                        # first copy can promote multiple later copies of the same photo.
                        if prior.duplicate_of:
                            continue
                        cosine = float(np.dot(vector, prior_vector))
                        kind = duplicate_kind(fp, prior_fp, result.candidate == prior.candidate, cosine)
                        if kind:
                            result.duplicate_of, result.duplicate_kind = prior.photo_id, kind
                            if result.accepted and not prior.accepted:
                                result.duplicate_of = result.duplicate_kind = None
                                prior.duplicate_of, prior.duplicate_kind = result.photo_id, kind
                                for member, _, _ in seen:
                                    if member.duplicate_of == prior.photo_id:
                                        member.duplicate_of = result.photo_id
                                        if kind == 'near':
                                            member.duplicate_kind = 'near'
                            else:
                                result.accepted = False
                            break
                        if cosine > .97:
                            result.similar_to = prior.photo_id
                    seen.append((result, fp, vector))
                    results.append(result)
            finally:
                for _, item in usable:
                    item[0].close()
            del prepared
        report = summarize(params.listing_id, results, self.vision.checklist, params.locale,
                           self.vision.version, bool(self.vision.calibration.get('validated')))
        report['model']['policy_version'] = POLICY_VERSION
        return report

    def close(self):
        self.pool.shutdown(wait=True)
        self.storage.close()
