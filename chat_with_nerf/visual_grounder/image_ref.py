from attrs import define
from PIL.Image import Image


@define
class ImageRef:
    rgb_address: str
    raw_image: Image
