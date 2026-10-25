"""Image loading and preprocessing shared by training, evaluation and the API.

Keeping one implementation here prevents train/serve skew.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "TIFF"}
MAX_PIXELS = 4096 * 4096


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a usable image."""


def load_image_bytes(data: bytes) -> Image.Image:
    """Decode and validate raw bytes into an RGB PIL image."""
    if not data:
        raise InvalidImageError("Empty file.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # detects truncated/corrupt files
        img = Image.open(io.BytesIO(data))  # verify() invalidates the object
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("File is not a readable image.") from exc
    if img.format not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"Unsupported format {img.format}. Use one of {sorted(ALLOWED_FORMATS)}."
        )
    if img.width * img.height > MAX_PIXELS:
        raise InvalidImageError("Image is too large.")
    return img.convert("RGB")


def resize_and_crop(img: Image.Image, image_size: int, crop_size: int) -> Image.Image:
    """Resize to a square of image_size, then center crop to crop_size."""
    img = img.resize((image_size, image_size), Image.BILINEAR)
    off = (image_size - crop_size) // 2
    return img.crop((off, off, off + crop_size, off + crop_size))


def to_normalized_array(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 CHW array normalised with ImageNet statistics."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).copy()


def preprocess(img: Image.Image, image_size: int, crop_size: int) -> np.ndarray:
    return to_normalized_array(resize_and_crop(img, image_size, crop_size))


def preprocess_mask(mask: Image.Image, image_size: int, crop_size: int) -> np.ndarray:
    """Ground-truth mask -> binary HxW uint8 array aligned with the image crop."""
    mask = mask.convert("L").resize((image_size, image_size), Image.NEAREST)
    off = (image_size - crop_size) // 2
    mask = mask.crop((off, off, off + crop_size, off + crop_size))
    return (np.asarray(mask) > 127).astype(np.uint8)
