import hashlib
import warnings
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def decode(data, max_pixels):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'} or getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('unsupported_image')
                if source.width * source.height > max_pixels:
                    raise ValueError('too_many_pixels')
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGB')
                original_size = image.size
                image.thumbnail((1024, 1024))
                return image, original_size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('invalid_image') from exc


def quality(image, original_size):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    # Fixed measurement scale makes the blur heuristic less resolution-dependent.
    scale = 512 / max(gray.shape)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark = float((gray < 20).mean())
    bright = float((gray > 245).mean())
    issues = []
    if min(original_size) < 256:
        issues.append('too_small')
    if sharpness < 20:
        issues.append('blurred')
    if dark > .85 and sharpness < 50:
        issues.append('underexposed')
    if bright > .85 and sharpness < 50:
        issues.append('overexposed')
    return issues


def fingerprint(data, image):
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(thumb)[:8, :8].flatten()[1:]
    return hashlib.sha256(data).hexdigest(), dct > np.median(dct), thumb / 255


def duplicate_kind(current, previous, same_category, cosine):
    if current[0] == previous[0]:
        return 'exact'
    distance = int(np.count_nonzero(current[1] != previous[1]))
    mse = float(np.mean((current[2] - previous[2]) ** 2))
    # Embeddings alone never suppress useful pictures of the same car.
    if same_category and distance <= 3 and mse < .003 and cosine > .985:
        return 'near'
    return None
