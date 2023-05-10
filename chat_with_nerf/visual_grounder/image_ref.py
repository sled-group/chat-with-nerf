from typing import Optional

import numpy as np
from attrs import define
from PIL.Image import Image


@define
class ImageRef:
    id: int
    clip_address: str
    rgb_address: str
    encoding: np.ndarray
    raw_image: Optional[Image] = None
