#!/usr/bin/env python
"""Visual Grounder."""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import mediapy as media
import numpy as np
import torch
from attrs import define
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from rich.console import Console

from chat_with_nerf import logger
from chat_with_nerf.visual_grounder.crop import CropData
from chat_with_nerf.visual_grounder.image_ref import ImageRef

CONSOLE = Console(width=120)


@define
class VisualGrounder:
    output_path: str
    """Output path."""
    camera_poses: dict
    """Determined camera poses."""
    lerf_pipeline: Pipeline | None
    """LERF pipeline."""
    downscale_factor: float = 1
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: int | None = None
    """Specifies number of rays per chunk during eval."""

    def set_positive_words(self, positive_word: str) -> bool:
        positive_word_list = [positive_word]

        try:
            self.lerf_pipeline.image_encoder.set_positives(positive_word_list)  # type: ignore
            return True
        except Exception:
            return False

    def taking_pictures(self, session_id: str) -> list[ImageRef]:
        install_checks.check_ffmpeg_installed()
        logger.info("picture taking process")
        camera_photo = self.camera_poses
        camera_path = get_path_from_json(camera_photo)
        # crop_data = get_crop_from_json(camera_path)
        camera_type = CameraType.PERSPECTIVE
        logger.info("sub picture taking process")
        result = self._taking_picture(
            session_id,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=["rgb", "relevancy_0"],
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=None,
            camera_type=camera_type,
        )
        return result

    def _taking_picture(
        self,
        session_id: str,
        cameras: Cameras,
        output_filename: str,
        rendered_output_names: list[str],
        crop_data: CropData | None = None,
        rendered_resolution_scaling_factor: float = 1.0,
        camera_type: CameraType = CameraType.PERSPECTIVE,
    ) -> list[ImageRef]:
        """Helper function to create 6 pictures of a given scene.

        Args:
            cameras: Cameras to render.
            output_filename: Name of the output file.
            rendered_output_names: List of outputs to visualise.
            crop_data: Crop data to apply to the rendered images.
            rendered_resolution_scaling_factor: Scaling factor to apply to the camera
            image resolution.
            camera_type: Camera projection format type.
        """
        CONSOLE.print("[bold green]Taking 6 images... ")
        cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
        cameras = cameras.to(self.lerf_pipeline.device)
        output_filepath_path = Path(output_filename) / session_id / "images"
        rgb_image_dir = output_filepath_path / "rgb"
        clip_image_dir = output_filepath_path / "clip"
        rgb_image_dir.mkdir(parents=True, exist_ok=True)
        clip_image_dir.mkdir(parents=True, exist_ok=True)

        result = {}
        return_result = []
        numpy_result = {}

        for camera_idx in range(cameras.size):
            aabb_box = None
            if crop_data is not None:
                bounding_box_min = crop_data.center - crop_data.scale / 2.0
                bounding_box_max = crop_data.center + crop_data.scale / 2.0
                aabb_box = SceneBox(
                    torch.stack([bounding_box_min, bounding_box_max]).to(
                        self.lerf_pipeline.device
                    )
                )
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, aabb_box=aabb_box
            )

            if crop_data is not None:
                with renderers.background_color_override_context(
                    crop_data.background_color.to(self.lerf_pipeline.device)
                ), torch.no_grad():
                    outputs = (
                        self.lerf_pipeline.model.get_outputs_for_camera_ray_bundle(
                            camera_ray_bundle
                        )
                    )
            else:
                with torch.no_grad():
                    outputs = (
                        self.lerf_pipeline.model.get_outputs_for_camera_ray_bundle(
                            camera_ray_bundle
                        )
                    )

            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(
                        f"Could not find {rendered_output_name} in the model outputs",
                        justify="center",
                    )
                    CONSOLE.print(
                        f"Please set --rendered_output_name to one of: {outputs.keys()}",
                        justify="center",
                    )
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                if output_image.shape[-1] == 1:
                    output_image = np.concatenate((output_image,) * 3, axis=-1)
                render_image.append(output_image)
            render_image_concat = np.concatenate(render_image, axis=1)

            rgb0 = render_image_concat[:, :512, :]
            relevancy0 = render_image_concat[:, 512:, :]
            grayscale_data = np.mean(relevancy0, axis=2)

            # Get the current timestamp
            now = datetime.datetime.now()
            # Format the timestamp as a string
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            # saving rgb
            rgb = "rgb" + str(camera_idx)
            # create file name
            rgb_filename = rgb + "_" + timestamp_str + ".png"
            result[rgb] = str(rgb_image_dir) + "/" + rgb_filename
            media.write_image(result[rgb], rgb0)

            # saving clip
            clip = "clip" + str(camera_idx)
            clip_file = clip + "_" + timestamp_str + ".npy"
            clip_filename = clip + "_" + timestamp_str + ".png"

            result[clip] = str(clip_image_dir) + "/" + clip_filename
            plt.imsave(result[clip], grayscale_data, cmap="turbo")

            # develop purpose
            numpy_result[clip] = str(clip_image_dir / clip_file)
            np.save(clip_image_dir / clip_file, relevancy0)

            imageRef = ImageRef(camera_idx, result[clip], result[rgb], relevancy0)
            return_result.append(imageRef)

        CONSOLE.print("[bold green]Finish taking 6 images... ")
        return return_result
