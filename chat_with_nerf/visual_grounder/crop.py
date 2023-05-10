#!/usr/bin/env python
"""Groundng Render.py."""
from __future__ import annotations

from typing import Any

import torch
from attrs import define
from torchtyping import TensorType  # type: ignore


@define
class CropData:
    """Data for cropping an image."""

    background_color: TensorType[3] = torch.Tensor([0.0, 0.0, 0.0])
    """Background color."""
    center: TensorType[3] = torch.Tensor([0.0, 0.0, 0.0])
    """Center of the crop."""
    scale: TensorType[3] = torch.Tensor([2.0, 2.0, 2.0])
    """Scale of the crop."""


def get_crop_from_json(camera_json: dict[str, Any]) -> CropData | None:
    """Load crop data from a camera path JSON.

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None

    bg_color = camera_json["crop"]["crop_bg_color"]

    return CropData(
        background_color=torch.Tensor(
            [bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]
        ),
        center=torch.Tensor(camera_json["crop"]["crop_center"]),
        scale=torch.Tensor(camera_json["crop"]["crop_scale"]),
    )
