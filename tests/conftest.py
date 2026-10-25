import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 80, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()
