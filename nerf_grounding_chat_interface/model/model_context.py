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
from nerf_grounding_chat_interface.model.util import rotate_x, rotate_y, rotate_z


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

        print("Initialize Blip2Captioner")
        blip2captioner = ModelContextManager.initiaze_blip_captioner()

        print("Initialize LERF pipeline")
        lerf_pipeline = ModelContextManager.initialize_lerf_pipeline(
            settings.load_config
        )
        settings = ModelContextManager.edit_settings(settings)

        print("Initialize visualGrounder")
        visual_grounder = VisualGrounder(
            settings.output_path, settings.camera_poses, lerf_pipeline
        )

        return ModelContext(visual_grounder, blip2captioner, lerf_pipeline)

    @staticmethod
    def initiaze_blip_captioner() -> Blip2Captioner:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
        )

        return Blip2Captioner(model, vis_processors)

    @staticmethod
    def initialize_lerf_pipeline(load_config: str) -> Pipeline:
        _, lerf_pipeline, _, _ = eval_setup(
            Path(load_config),
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )

        return lerf_pipeline

    @staticmethod
    def edit_settings(settings: Settings):
        # List all files in the directory
        all_files = [
            f
            for f in os.listdir(settings.data_path)
            if os.path.isfile(os.path.join(settings.data_path, f))
        ]
        # Sort the files alphabetically
        all_files.sort()
        # Get the first file in the sorted list
        first_file = all_files[0] if all_files else None

        if first_file is not None:
            file_path = settings.data_path + "/" + first_file

        # Replace 'file_path.json' with the actual path to your JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        camera_to_world_matrix = np.zeros((4, 4))

        prefix = "t_"

        for i in range(3):
            for j in range(4):
                camera_to_world_matrix[i][j] = data[prefix + str(i) + str(j)]

        camera_to_world_matrix[3][3] = 1
        camera_to_world_matrix[0][3] = 0.3
        camera_to_world_matrix[1][3] = 0.04
        camera_to_world_matrix[2][3] = 0.15

        c2w = rotate_y(-60, camera_to_world_matrix)
        c2w = rotate_z(-90, c2w)
        c2w_new = c2w.reshape(16, 1)

        settings.camera_poses["camera_path"][0]["camera_to_world"] = c2w_new.tolist()
        for i in range(1, len(settings.camera_poses["camera_path"])):
            c2w = rotate_y(60, c2w)
            c2w_new = rotate_x(-20, c2w)
            c2w_new = c2w_new.reshape(16, 1)
            settings.camera_poses["camera_path"][i][
                "camera_to_world"
            ] = c2w_new.tolist()
            settings.camera_poses["camera_path"][i]["fov"] = 90

        return settings
