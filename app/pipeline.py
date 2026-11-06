import hashlib

from app.events import REQUIRED
from app.schemas import EvaluateParams, Photo
from app.state import ContractError


class RetryableError(RuntimeError):
    pass


def apply_manual_representatives(photos, references):
    """One usable, manually corrected representative per existing duplicate group."""
    groups = {}
    # Work on copies: model output must remain an unmodified diagnostic record.
    output = [dict(photo) for photo in photos]
    for photo in output:
        groups.setdefault(photo.get('duplicate_of') or photo['photo_id'], []).append(photo)
    for root, members in groups.items():
        corrected = [p for p in members
                     if references[p['photo_id']].manual_category is not None
                     and p.get('quality') == 'good' and not p.get('error')]
        if not corrected:
            continue
        chosen = next((p for p in corrected if p['photo_id'] == root), corrected[0])
        chosen_kind = chosen.get('duplicate_kind')
        for photo in members:
            if photo is chosen:
                photo['duplicate_of'] = photo['duplicate_kind'] = None
            else:
                photo['accepted'] = False
                photo['duplicate_of'] = chosen['photo_id']
                photo['duplicate_kind'] = ('near' if chosen_kind == 'near'
                                           or photo.get('duplicate_kind') == 'near' else 'exact')
    return output


class Pipeline:
    def __init__(self, settings, advisor, state):
        self.settings, self.advisor, self.state = settings, advisor, state

    def envelope(self, event, status):
        result_id = hashlib.sha256(
            f'{event.listing_id}:{event.listing_revision}'.encode()).hexdigest()
        return dict(schema_version=1, result_id=result_id, event_id=event.event_id,
                    correlation_id=event.correlation_id, listing_id=event.listing_id,
                    listing_revision=event.listing_revision, status=status,
                    total_required=len(REQUIRED), recognized=0, missing=None,
                    uncertain=[], photos=[])

    def validate(self, event):
        photos = event.photos if event.photos is not None else ([event.photo] if event.photo else [])
        if len(photos) > self.settings.max_photos:
            raise ContractError('too_many_photos')
        for photo in photos:
            if photo.bucket != self.settings.s3_bucket:
                raise ContractError('bucket_not_allowed')
            try:
                self.advisor.storage.validate_key(event.listing_id, photo.object_key)
            except Exception as exc:
                raise ContractError('object_key_not_allowed') from exc

    def analyze(self, event):
        if event.event_type != 'photoset.finalized':
            return self.envelope(event, 'processing')
        params = EvaluateParams(listing_id=event.listing_id, photos=[
            Photo(**p.model_dump(include={'photo_id', 'object_key', 'version_id', 'etag'}))
            for p in event.photos])
        with self.advisor.lock:
            report = self.advisor.evaluate(params)
        photos = report['photo_results']
        expected = {p.photo_id for p in event.photos}
        if len(photos) != len(expected) or {p['photo_id'] for p in photos} != expected:
            raise RetryableError('incomplete_analysis_response')
        if any(p.get('error') in {'s3_unavailable', 'inference_failed'} for p in photos):
            raise RetryableError('analysis_temporarily_unavailable')
        by_id = {p.photo_id: p for p in event.photos}
        photos = apply_manual_representatives(photos, by_id)
        present, uncertain, output = set(), [], []
        for photo in photos:
            reference = by_id[photo['photo_id']]
            category = photo['category']
            requested = reference.manual_category
            rejection = None
            if requested is not None:
                if photo.get('error'):
                    rejection = 'photo_error'
                elif photo.get('quality') != 'good':
                    rejection = 'quality_not_good'
                elif photo.get('duplicate_of'):
                    rejection = 'duplicate_photo'
            manual = requested if rejection is None else None
            if manual is not None:
                category = manual
                accepted = True
            else:
                accepted = (photo['accepted'] and category in REQUIRED
                            and photo.get('quality') == 'good' and not photo.get('error')
                            and not photo.get('duplicate_of'))
            if accepted:
                present.add(category)
            elif not photo.get('duplicate_of'):
                uncertain.append(photo['photo_id'])
            output.append(dict(image_id=photo['photo_id'],
                               category=category if accepted else (
                                   'uncertain' if photo['recognition'] == 'uncertain' else 'unknown'),
                               candidate=photo.get('candidate') or photo['category'],
                               quality=photo.get('quality', 'unavailable'),
                               quality_issues=photo.get('quality_issues', []),
                               duplicate_of=photo.get('duplicate_of'),
                               confidence=None if manual else photo.get('confidence'),
                               similarity=photo.get('similarity'), accepted=accepted,
                               score=photo.get('score'), score_kind=photo.get('score_kind'),
                               manual_category=requested, manual_applied=manual is not None,
                               manual_rejection_reason=rejection,
                               source='manual' if manual else 'model', error=photo.get('error')))
        failed = bool(photos) and all(p.get('error') for p in photos)
        result = self.envelope(event, 'failed' if failed else ('needs_review' if uncertain else 'completed'))
        result.update(recognized=len(present), missing=[c for c in REQUIRED if c not in present],
                      uncertain=uncertain, photos=output, model=report['model'])
        return result
