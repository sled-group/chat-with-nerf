from typing import Optional

from attrs import define
from PIL.Image import Image


@define
class ImageRef:
    id: int
    rgb_address: str
    raw_image: Optional[Image] = None
