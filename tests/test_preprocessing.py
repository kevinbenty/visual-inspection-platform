import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (InvalidImageError, load_image_bytes, preprocess,
                               preprocess_mask)


def test_preprocess_shape_and_dtype(png_bytes):
    arr = preprocess(load_image_bytes(png_bytes), 256, 224)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32


def test_grayscale_converted_to_rgb():
    buf = io.BytesIO()
    Image.new("L", (32, 32), 128).save(buf, "PNG")
    assert load_image_bytes(buf.getvalue()).mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_rejects_bad_bytes(data):
    with pytest.raises(InvalidImageError):
        load_image_bytes(data)


def test_rejects_unsupported_format():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, "GIF")
    with pytest.raises(InvalidImageError):
        load_image_bytes(buf.getvalue())


def test_mask_aligned_with_crop():
    m = np.zeros((256, 256), np.uint8)
    m[100:150, 100:150] = 255
    out = preprocess_mask(Image.fromarray(m), 256, 224)
    assert out.shape == (224, 224)
    assert set(np.unique(out)) == {0, 1}
    assert out[84:134, 84:134].all()  # shifted by (256-224)/2 = 16
