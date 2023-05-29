from typing import Optional

from attrs import define
from PIL.Image import Image


@define
class ImageRef:
    # id: int
    # clip_address: str
    rgb_address: str
    # encoding: np.ndarray
    rgb_image: Optional[Image] = None
