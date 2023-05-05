#!/usr/bin/env python
"""
Visual Grounder
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt  # type: ignore
import mediapy as media
import numpy as np
import torch
from rich.console import Console
from nerfstudio.cameras.camera_paths import (
    get_path_from_json,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerf_grounding_chat_interface.visual_grounder.image_ref import ImageRef
from nerf_grounding_chat_interface.visual_grounder.crop import (
    CropData,
    get_crop_from_json,
)

CONSOLE = Console(width=120)


class VisualGrounder:
    def __init__(self, load_config, output_path, camera_poses):
        self.load_config = load_config
        """Path to config YAML file."""
        self.output_path = output_path
        """Output path."""
        self.camera_poses = camera_poses
        """Determined camera poses"""
        self.downscale_factor = 1
        """Scaling factor to apply to the camera image resolution."""
        self.eval_num_rays_per_chunk = None
        """Specifies number of rays per chunk during eval."""

    def construct_pipeline(self) -> Pipeline:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            Path(
                "/workspace/inthewild/outputs/bigexample/lerf/2023-04-22_022439/config.yml"
            ),
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )

        return pipeline

    def set_positive_words(self, pipeline: Pipeline, positive_word: str) -> bool:
        positive_word_list = [positive_word]

        try:
            pipeline.image_encoder.set_positives(positive_word_list)
            return True
        except Exception:
            return False

    def taking_pictures(self, pipeline: Pipeline):
        install_checks.check_ffmpeg_installed()
        print("picture taking process")
        camera_photo = self.camera_poses
        camera_path = get_path_from_json(camera_photo)
        # crop_data = get_crop_from_json(camera_path)
        camera_type = CameraType.PERSPECTIVE
        print("sub picture taking process")
        result = self._taking_picture(
            pipeline,
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
        pipeline: Pipeline,
        cameras: Cameras,
        output_filename: str,
        rendered_output_names: List[str],
        crop_data: Optional[CropData] = None,
        rendered_resolution_scaling_factor: float = 1.0,
        camera_type: CameraType = CameraType.PERSPECTIVE,
    ) -> list[ImageRef]:
        """Helper function to create 6 pictures of a given scene.

        Args:
            pipeline: Pipeline to evaluate with.
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
        cameras = cameras.to(pipeline.device)
        output_filepath_path = Path(output_filename)
        rgb_image_dir = output_filepath_path.parent / "rgb"
        clip_image_dir = output_filepath_path.parent / "clip"
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
                        pipeline.device
                    )
                )
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, aabb_box=aabb_box
            )

            if crop_data is not None:
                with renderers.background_color_override_context(
                    crop_data.background_color.to(pipeline.device)
                ), torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle
                    )
            else:
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle
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
            render_image = np.concatenate(render_image, axis=1)

            rgb0 = render_image[:, :512, :]
            relevancy0 = render_image[:, 512:, :]
            grayscale_data = np.mean(relevancy0, axis=2)

            # saving rgb
            rgb = "rgb" + str(camera_idx)
            result[rgb] = str(rgb_image_dir) + "/" + str(f"{camera_idx:03d}.png")
            media.write_image(rgb_image_dir / f"{camera_idx:03d}.png", rgb0)

            # saving clip
            clip = "clip" + str(camera_idx)
            clip_file = clip + ".npy"
            result[clip] = str(clip_image_dir) + "/" + str(f"{camera_idx:03d}.png")
            # media.write_image(clip_image_dir / f"{camera_idx:03d}.png", rgb0)
            plt.imsave(result[clip], grayscale_data, cmap="turbo")

            numpy_result[clip] = str(clip_image_dir / clip_file)
            np.save(clip_image_dir / clip_file, relevancy0)

            imageRef = ImageRef(camera_idx, result[clip], result[rgb], relevancy0)
            return_result.append(imageRef)

        CONSOLE.print("[bold green]Finish taking 6 images... ")
        return return_result
