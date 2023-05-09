import os
import sys
from pathlib import Path
from typing import Optional
import json
import numpy as np

import torch
from attrs import define
from lavis.models import load_model_and_preprocess  # type: ignore
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup

from nerf_grounding_chat_interface.visual_grounder.blip2_caption import Blip2Captioner
from nerf_grounding_chat_interface.visual_grounder.settings import Settings
from nerf_grounding_chat_interface.visual_grounder.visual_grounder import VisualGrounder


@define
class ModelContext:
    visualGrounder: VisualGrounder
    blip2captioner: Blip2Captioner
    pipeline: Pipeline


class ModelContextManager:
    model_context: Optional[ModelContext] = None

    @classmethod
    def get_model_context(cls):
        if cls.model_context is None:
            cls.model_context = ModelContextManager.initialize_model_context()
        return cls.model_context

    @staticmethod
    def initialize_model_context() -> ModelContext:
        # Get the absolute path of the project's root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Add the project's root directory to sys.path
        sys.path.append(project_root)

        settings = Settings()

        print("load_config: ", settings.load_config)
        print("output_path: ", settings.output_path)

        print("Initialize Blip2Captioner")
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
        )
        blip2captioner = Blip2Captioner(model, vis_processors)

        print("Initialize LERF pipeline")
        _, lerf_pipeline, _, _ = eval_setup(
            Path(
                "/workspace/inthewild/outputs/bigexample/lerf/2023-04-22_022439/config.yml"
            ),
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )

        # editing the camera poses
        # Replace 'your_directory_path' with the actual path to the directory
        # TODO: change this to a more general path
        directory_path = "/workspace/inthewild/bigexample/Apr7at4-46PM-poly/keyframes/corrected_cameras"

        # List all files in the directory
        all_files = [
            f
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]

        # Sort the files alphabetically
        all_files.sort()

        # Get the first file in the sorted list
        first_file = all_files[0] if all_files else None

        if first_file is not None:
            file_path = directory_path + "/" + first_file

        # Replace 'file_path.json' with the actual path to your JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        camera_to_world_matrix = np.zeros((4, 4))

        prefix = "t_"

        for i in range(3):
            for j in range(4):
                # print(prefix + str(i) + str(j))
                # print(data[prefix + str(i) + str(j)])
                camera_to_world_matrix[i][j] = data[prefix + str(i) + str(j)]

        camera_to_world_matrix[3][3] = 1

        camera_pose_preset = settings.camera_poses

        angle_degrees = 60
        angle_radians = np.radians(angle_degrees)
        rotation_matrix_x = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
                [0, np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 0, 1],
            ]
        )

        rotation_matrix_y = np.array(
            [
                [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
                [0, 0, 0, 1],
            ]
        )

        rotation_matrix_z = np.array(
            [
                [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        c2w = camera_to_world_matrix
        c2w_new = c2w.reshape(16, 1)
        settings.camera_poses["camera_path"][0]["camera_to_world"] = c2w_new.tolist()
        for i in range(1, len(settings.camera_poses["camera_path"])):
            c2w = c2w @ rotation_matrix_z
            c2w_new = c2w.reshape(16, 1)
            settings.camera_poses["camera_path"][i][
                "camera_to_world"
            ] = c2w_new.tolist()

        print("Initialize visualGrounder")
        visual_grounder = VisualGrounder(
            settings.output_path, settings.camera_poses, lerf_pipeline
        )

        return ModelContext(visual_grounder, blip2captioner, lerf_pipeline)
