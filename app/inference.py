import hashlib
import json

import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_tensor
from transformers import CLIPModel, CLIPProcessor

from app.policy import validate_calibration


MODEL_ID = 'openai/clip-vit-base-patch32'
MODEL_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'


def load_vision(settings):
    if settings.cv_backend == 'trained':
        from app.trained import TrainedVision
        return TrainedVision(settings)
    return Vision(settings)


def vehicle_visibility(candidate, cars, size):
    exterior = candidate in {'front', 'rear', 'left_side', 'right_side',
                             'front_left', 'front_right', 'rear_left', 'rear_right'}
    if not exterior:
        return []
    if not cars:
        return ['car_not_detected']
    if len(cars) > 1:
        return ['multiple_cars']
    x1, y1, x2, y2 = cars[0]
    w, h = size
    issues = []
    if x1 < .015*w or y1 < .015*h or x2 > .985*w or y2 > .985*h:
        issues.append('possible_crop')
    if (x2-x1)*(y2-y1)/(w*h) < .15:
        issues.append('car_too_small')
    return issues


class Vision:
    """Local-only inference. Model files are provisioned before service startup."""

    def __init__(self, settings):
        self.settings = settings
        torch.set_num_threads(settings.cpu_threads)
        self.device = torch.device(settings.device)
        self.checklist = json.loads(settings.checklist_path.read_text(encoding='utf-8'))
        self.labels = self.checklist['classes']
        self.names = list(self.labels)
        if len(self.labels) < 2 or any(
                not isinstance(prompts, list) or not prompts
                or any(not isinstance(p, str) or not p.strip() for p in prompts)
                for prompts in self.labels.values()):
            raise ValueError('taxonomy requires at least two classes with nonempty prompts')
        ids = [c['id'] for c in self.checklist['categories']]
        if (len(ids) != len(set(ids))
                or any(k not in self.labels or v not in ids
                       for k, v in self.checklist['coverage'].items())):
            raise ValueError('invalid checklist coverage')
        manifest = json.loads((settings.model_dir / 'manifest.json').read_text())
        if manifest['revision'] != MODEL_REVISION or manifest['model_id'] != MODEL_ID:
            raise ValueError('unexpected model revision')
        root = settings.model_dir.parent.resolve()
        runtime_files = [p for p in settings.model_dir.iterdir()
                         if p.suffix in {'.json', '.txt', '.bin', '.safetensors'} and p.name != 'manifest.json']
        runtime_files.append(settings.detector_path)
        for path in runtime_files:
            if (not path.resolve().is_relative_to(root)
                    or path.resolve().relative_to(root).as_posix() not in manifest['sha256']):
                raise ValueError('runtime artifact missing from manifest')
        for relative, expected in manifest['sha256'].items():
            file = (settings.model_dir.parent / relative).resolve()
            if not file.is_relative_to(settings.model_dir.parent.resolve()):
                raise ValueError('invalid artifact path')
            with file.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('model artifact checksum mismatch')
        self.version = f'{MODEL_ID}@{MODEL_REVISION}'
        self.taxonomy_hash = hashlib.sha256(
            json.dumps(self.checklist, sort_keys=True).encode()).hexdigest()
        self.calibration = validate_calibration(
            json.loads(settings.calibration_path.read_text()), self.version, self.taxonomy_hash, self.names)
        self.processor = CLIPProcessor.from_pretrained(settings.model_dir, local_files_only=True, use_fast=False)
        self.model = CLIPModel.from_pretrained(settings.model_dir, local_files_only=True)
        self.model.to(self.device).eval()
        self.detector = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.detector.load_state_dict(torch.load(settings.detector_path, map_location='cpu', weights_only=True))
        self.detector.to(self.device).eval()
        with torch.inference_mode():
            texts = [prompt for prompts in self.labels.values() for prompt in prompts]
            inputs = self.processor(text=texts, return_tensors='pt', padding=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
            prototypes, offset = [], 0
            for prompts in self.labels.values():
                prototypes.append(features[offset:offset + len(prompts)].mean(dim=0))
                offset += len(prompts)
            self.prototypes = torch.nn.functional.normalize(torch.stack(prototypes), dim=-1)
        # Text tower is no longer needed for requests.
        del self.model.text_model

    @torch.inference_mode()
    def predict(self, images):
        if not images:
            return []
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        vectors = torch.nn.functional.normalize(self.model.get_image_features(**inputs), dim=-1)
        similarities = (vectors @ self.prototypes.T).cpu().numpy()
        detections = self.detector([to_tensor(im).to(self.device) for im in images])
        results = []
        for image, scores, vector, detection in zip(images, similarities, vectors.cpu().numpy(), detections):
            order = np.argsort(scores)[::-1]
            idx, second = int(order[0]), int(order[1])
            candidate = self.names[idx]
            score = float(scores[idx])
            margin = score - float(scores[second])
            policy = self.calibration.get('classes', {}).get(candidate)
            calibrated = bool(self.calibration.get('validated') and policy)
            passes = bool(calibrated and score >= policy['min_similarity'] and margin >= policy['min_margin'])
            unknown = candidate in {'other', 'unknown'} or score < self.calibration['unknown_similarity']
            recognition = 'unknown' if unknown else ('known' if passes else 'uncertain')
            cars = []
            for box, label, probability in zip(detection['boxes'], detection['labels'], detection['scores']):
                if int(label) == 3 and float(probability) >= .5:
                    cars.append([float(x) for x in box.cpu()])
            cars.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            issues = vehicle_visibility(candidate, cars, image.size)
            # Occlusion and body inspection suitability need domain labels; do not invent a detector.
            return_category = 'unknown' if unknown else candidate
            results.append(dict(category=return_category, candidate=candidate, similarity=score,
                                margin=margin, recognition=recognition, confidence=None,
                                confidence_kind='similarity_thresholds' if calibrated else 'not_calibrated',
                                quality_issues=issues,
                                alternatives=[{'category': self.names[int(i)], 'similarity': float(scores[i])}
                                              for i in order[:3]],
                                detection={'car_count': len(cars), 'primary_box': cars[0] if cars else None,
                                           'occlusion_assessed': False}, embedding=vector))
        return results
