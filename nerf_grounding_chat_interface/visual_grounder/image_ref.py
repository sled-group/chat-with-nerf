import numpy as np
from attrs import define


@define
class ImageRef:
    id: int
    clip_address: str
    rgb_address: str
    encoding: np.ndarray
