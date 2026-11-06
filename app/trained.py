"""Small supervised view classifier. Runtime never downloads model weights."""
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.transforms.functional import to_tensor

from app.events import REQUIRED
from app.evaluation import wilson_lower


LABELS = (*REQUIRED, 'other')
ARCHITECTURE = 'mobilenet_v3_small_spatial_v1'
FEATURE_DIM = 576 * 5
CLIP_ARCHITECTURE = 'clip_vit_b32_spatial_supervised_v1'
CLIP_FEATURE_DIM = 512 * 5
FUSED_ARCHITECTURE = 'mobilenet_clip_spatial_supervised_v1'
FUSED_FEATURE_DIM = FEATURE_DIM + CLIP_FEATURE_DIM
RESIDUAL_FUSED_ARCHITECTURE = 'mobilenet_clip_spatial_residual_supervised_v1'
DINO_ARCHITECTURE = 'dinov2_small_spatial_supervised_v1'
DINO_FEATURE_DIM = 384 * 5
TRIPLE_ARCHITECTURE = 'mobilenet_clip_dinov2_spatial_supervised_v1'
TRIPLE_FEATURE_DIM = FUSED_FEATURE_DIM + DINO_FEATURE_DIM


class ScoreGate(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    min_score: float = Field(ge=0, le=1)
    min_margin: float = Field(ge=0, le=1)


class SupervisedPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    validated: bool = Field(default=False, strict=True)
    taxonomy_hash: str | None = None
    classes: dict[str, ScoreGate] = Field(default_factory=dict)

    @model_validator(mode='after')
    def known_classes(self):
        if set(self.classes) - set(REQUIRED):
            raise ValueError('acceptance gates must belong to the twelve required views')
        if self.validated and not self.classes:
            raise ValueError('validated supervised policy has no class gates')
        return self


def load_policy(manifest, weight_hash, taxonomy_hash):
    policy = SupervisedPolicy.model_validate(manifest.get('calibration', {}))
    if policy.validated:
        approval = manifest.get('heldout_approval')
        if (not isinstance(approval, dict) or approval.get('model_sha256') != weight_hash
                or approval.get('dataset_id') != manifest.get('dataset_id')
                or not approval.get('test_manifest_sha256') or policy.taxonomy_hash != taxonomy_hash):
            raise ValueError('supervised release lacks matching held-out approval')
        policy_hash = hashlib.sha256(json.dumps(policy.model_dump(), sort_keys=True).encode()).hexdigest()
        if approval.get('policy_sha256') != policy_hash:
            raise ValueError('supervised acceptance policy changed after approval')
        for category in policy.classes:
            evidence = approval.get('classes', {}).get(category, {})
            values = [evidence.get(k) for k in ('tp', 'fp', 'fn', 'tn')]
            if any(type(n) is not int or n < 0 for n in values):
                raise ValueError('invalid held-out class evidence')
            tp, fp, fn, tn = values
            lower = wilson_lower(tp, tp+fp)
            if tp+fn < 30 or tn+fp < 30 or lower is None or lower < .95:
                raise ValueError('insufficient held-out evidence for automatic acceptance')
    return policy.model_dump()


def prepare_image(image, clip=False):
    padded = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC,
                         color=(127, 127, 127))
    try:
        tensor = to_tensor(padded)
    finally:
        padded.close()
    mean = tensor.new_tensor([.48145466, .4578275, .40821073] if clip else [.485, .456, .406])[:, None, None]
    std = tensor.new_tensor([.26862954, .26130258, .27577711] if clip else [.229, .224, .225])[:, None, None]
    return (tensor - mean) / std


class AngleNet(nn.Module):
    architecture = ARCHITECTURE
    preprocessing = 'letterbox_rgb_imagenet_v1'
    prepare_image = staticmethod(prepare_image)
    def __init__(self, weights=None):
        super().__init__()
        self.features = mobilenet_v3_small(weights=weights).features
        self.classifier = nn.Linear(FEATURE_DIM, len(LABELS))

    def embed(self, images):
        maps = self.features(images)
        global_features = nn.functional.adaptive_avg_pool2d(maps, 1).flatten(1)
        spatial = nn.functional.adaptive_avg_pool2d(maps, 2).flatten(1)
        return nn.functional.normalize(torch.cat((global_features, spatial), dim=1), dim=1)

    def forward(self, images):
        return self.classifier(self.embed(images))


class ClipAngleNet(nn.Module):
    """A single frozen visual encoder with global and spatial supervised features."""
    architecture = CLIP_ARCHITECTURE
    preprocessing = 'letterbox_rgb_clip_v1'

    def __init__(self, weights=None, encoder=None):
        super().__init__()
        if encoder is not None:
            self.features = encoder
        else:
            from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
            if weights is not None:
                self.features = CLIPVisionModelWithProjection.from_pretrained(weights, local_files_only=True)
            else:
                self.features = CLIPVisionModelWithProjection(CLIPVisionConfig(
                    hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                    num_attention_heads=12, image_size=224, patch_size=32,
                    projection_dim=512, hidden_act='quick_gelu', layer_norm_eps=1e-5))
        self.classifier = nn.Linear(CLIP_FEATURE_DIM, len(LABELS))

    @staticmethod
    def prepare_image(image):
        return prepare_image(image, clip=True)

    def embed(self, images):
        output = self.features(pixel_values=images)
        tokens = self.features.vision_model.post_layernorm(output.last_hidden_state[:, 1:])
        patches = self.features.visual_projection(tokens).transpose(1, 2).reshape(-1, 512, 7, 7)
        spatial = nn.functional.adaptive_avg_pool2d(patches, 2).flatten(1)
        return nn.functional.normalize(torch.cat((output.image_embeds, spatial), dim=1), dim=1)

    def forward(self, images):
        return self.classifier(self.embed(images))


class FusedAngleNet(nn.Module):
    architecture = FUSED_ARCHITECTURE
    preprocessing = 'letterbox_rgb_imagenet_clip_v1'

    def __init__(self, weights=None, mobile=None, clip=None):
        super().__init__()
        self.features = nn.ModuleDict({'mobile': mobile if mobile is not None else AngleNet(),
            'clip': clip if clip is not None else ClipAngleNet(weights=weights)})
        self.classifier = nn.Linear(FUSED_FEATURE_DIM, len(LABELS))

    @staticmethod
    def prepare_image(image):
        return torch.stack((prepare_image(image), prepare_image(image, clip=True)))

    def embed(self, images):
        mobile = self.features['mobile'].embed(images[:, 0])
        clip = self.features['clip'].embed(images[:, 1])
        return nn.functional.normalize(torch.cat((mobile, clip), dim=1), dim=1)

    def forward(self, images):
        return self.classifier(self.embed(images))


class DinoAngleNet(nn.Module):
    architecture = DINO_ARCHITECTURE
    preprocessing = 'letterbox_rgb_dinov2_imagenet_v1'
    prepare_image = staticmethod(prepare_image)

    def __init__(self, weights=None, encoder=None):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        self.features = (encoder if encoder is not None else
            Dinov2Model.from_pretrained(weights, local_files_only=True) if weights is not None else
            Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                image_size=518, patch_size=14, layer_norm_eps=1e-6, layerscale_value=1.,
                mlp_ratio=4, use_swiglu_ffn=False)))
        self.classifier = nn.Linear(DINO_FEATURE_DIM, len(LABELS))

    def embed(self, images):
        tokens = self.features(pixel_values=images).last_hidden_state
        side = images.shape[-1] // 14
        patches = tokens[:, 1:].transpose(1, 2).reshape(-1, 384, side, side)
        spatial = nn.functional.adaptive_avg_pool2d(patches, 2).flatten(1)
        return nn.functional.normalize(torch.cat((tokens[:, 0], spatial), dim=1), dim=1)

    def forward(self, images):
        return self.classifier(self.embed(images))


class TripleAngleNet(FusedAngleNet):
    architecture = TRIPLE_ARCHITECTURE

    def __init__(self, fused=None, dino=None):
        nn.Module.__init__(self)
        self.features = nn.ModuleDict({'fused': fused if fused is not None else FusedAngleNet(),
                                      'dino': dino if dino is not None else DinoAngleNet()})
        self.classifier = nn.Linear(TRIPLE_FEATURE_DIM, len(LABELS))

    def embed(self, images):
        fused = self.features['fused'].embed(images)
        dino = self.features['dino'].embed(images[:, 0])
        # The fused vector already contains two equally normalized encoders.
        return nn.functional.normalize(torch.cat((fused * (2 ** .5), dino), dim=1), dim=1)


class ResidualHead(nn.Module):
    """Preserve a frozen linear baseline, then learn a small nonlinear correction."""
    def __init__(self, features=FUSED_FEATURE_DIM, classes=len(LABELS)):
        super().__init__()
        self.base = nn.Linear(features, classes)
        self.base.requires_grad_(False)
        self.delta = nn.Sequential(nn.LayerNorm(features, elementwise_affine=False),
            nn.Linear(features, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, classes))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, features):
        return self.base(features) + self.delta(features)


class ResidualFusedAngleNet(FusedAngleNet):
    architecture = RESIDUAL_FUSED_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.classifier = ResidualHead()


def restore_runtime_buffers(model):
    """Recreate CLIP's deterministic position IDs, absent from checkpoint state."""
    for encoder in model.modules():
        if isinstance(encoder, ClipAngleNet):
            embeddings = encoder.features.vision_model.embeddings
            embeddings.position_ids = torch.arange(embeddings.num_positions,device='cpu').expand((1,-1))
    if any(value.is_meta for value in (*model.parameters(),*model.buffers())):
        raise ValueError('checkpoint left an unmaterialized model tensor')


class TrainedVision:
    def __init__(self, settings, *, memory_efficient=True):
        torch.set_num_threads(settings.cpu_threads)
        self.device = torch.device(settings.device)
        root = settings.trained_model_dir
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        architecture = manifest.get('architecture')
        models = {ARCHITECTURE: AngleNet, CLIP_ARCHITECTURE: ClipAngleNet,
                  FUSED_ARCHITECTURE: FusedAngleNet,
                  RESIDUAL_FUSED_ARCHITECTURE: ResidualFusedAngleNet,
                  DINO_ARCHITECTURE: DinoAngleNet, TRIPLE_ARCHITECTURE: TripleAngleNet}
        if architecture not in models or manifest.get('labels') != list(LABELS):
            raise ValueError('unsupported supervised model architecture/taxonomy')
        if manifest.get('input_size') != 224 or manifest.get('preprocessing') != models[architecture].preprocessing:
            raise ValueError('supervised preprocessing does not match training')
        weights = root / 'weights.pt'
        with weights.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest['sha256']:
            raise ValueError('supervised weights checksum mismatch')
        self.checklist = json.loads(settings.checklist_path.read_text(encoding='utf-8'))
        if {c['id'] for c in self.checklist['categories']} != set(REQUIRED):
            raise ValueError('supervised model requires the twelve-category checklist')
        self.names = list(LABELS)
        self.taxonomy_hash = hashlib.sha256(json.dumps(self.checklist, sort_keys=True).encode()).hexdigest()
        self.version = f'{architecture}@{digest}'
        self.calibration = load_policy(manifest, digest, self.taxonomy_hash)
        if memory_efficient:
            # Instantiate shapes only; avoid allocating a random encoder plus a
            # second full copy of the checkpoint during startup.
            with torch.device('meta'):
                self.model = models[architecture]()
            state = torch.load(weights, weights_only=True, map_location='cpu', mmap=True)
            self.model.load_state_dict(state, assign=True)
            restore_runtime_buffers(self.model)
        else:
            self.model = models[architecture]()
            self.model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, images):
        if not images:
            return []
        inputs = torch.stack([self.model.prepare_image(image) for image in images]).to(self.device)
        embeddings = self.model.embed(inputs)
        scores = self.model.classifier(embeddings).softmax(dim=1).cpu().numpy()
        outputs = []
        for probabilities, vector in zip(scores, embeddings.cpu().numpy()):
            order = np.argsort(probabilities)[::-1]
            candidate = LABELS[int(order[0])]
            score = float(probabilities[order[0]])
            margin = score - float(probabilities[order[1]])
            gate = self.calibration.get('classes', {}).get(candidate)
            accepted = bool(self.calibration.get('validated') and gate
                            and score >= gate['min_score'] and margin >= gate['min_margin'])
            recognition = 'unknown' if candidate == 'other' else ('known' if accepted else 'uncertain')
            outputs.append(dict(category='unknown' if candidate == 'other' else candidate,
                candidate=candidate, confidence=None,
                confidence_kind='validated_score_thresholds' if self.calibration['validated'] else 'not_calibrated',
                score=score, score_kind='softmax', margin=margin, recognition=recognition,
                quality_issues=[], detection={'occlusion_assessed': False},
                alternatives=[{'category': LABELS[int(i)], 'score': float(probabilities[i])}
                              for i in order[:3]], embedding=vector))
        return outputs
