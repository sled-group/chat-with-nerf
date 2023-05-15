import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from attrs import define
from lavis.models import load_model_and_preprocess  # type: ignore
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup

from chat_with_nerf import logger
from chat_with_nerf.model.scene_config import SceneConfig
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.blip2_caption import Blip2Captioner
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


@define
class ModelContext:
    scene_configs: dict[str, SceneConfig]
    visual_grounder: dict[str, VisualGrounder]
    pipeline: dict[str, Pipeline]
    blip2captioner: Blip2Captioner


class ModelContextManager:
    model_context: Optional[ModelContext] = None

    @classmethod
    def get_model_context(cls) -> ModelContext:
        if cls.model_context is None:
            cls.model_context = ModelContextManager.initialize_model_context()
        return cls.model_context

    @staticmethod
    def initialize_model_context() -> ModelContext:
        # Get the absolute path of the project's root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Add the project's root directory to sys.path
        sys.path.append(project_root)

        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(Settings.data_path)

        logger.info("Initialize Blip2Captioner")
        blip2captioner = ModelContextManager.initiaze_blip_captioner()

        logger.info("Initialize LERF pipelines and visualGrounder for all scenes")
        pipeline = {}
        visual_grounder_ins = {}
        initial_dir = os.getcwd()
        for i, (scene_name, scene_config) in enumerate(scene_configs.items()):
            if i == 0:
                # LERF's implementation requires to find output directory
                os.chdir(Settings.data_path + "/" + scene_name)
                lerf_pipeline = ModelContextManager.initialize_lerf_pipeline(
                    scene_config.load_lerf_config
                )
                pipeline[scene_name] = lerf_pipeline
                visual_grounder_ins[scene_name] = VisualGrounder(
                    Settings.output_path, scene_config.camera_poses, lerf_pipeline
                )

        # move back the current directory
        os.chdir(initial_dir)
        return ModelContext(
            scene_configs, visual_grounder_ins, pipeline, blip2captioner
        )

    @staticmethod
    def search_scenes(path: str) -> dict[str, SceneConfig]:
        scenes = {}
        subdirectories = [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]

        for subdirectory in subdirectories:
            scene_path = (Path(path) / subdirectory / subdirectory).with_suffix(".yaml")
            with open(scene_path) as f:
                data = yaml.safe_load(f)
            scene = SceneConfig(
                data["load_lerf_config"], data["camera_path"], data["camera_poses"]
            )
            scenes[subdirectory] = scene
        return scenes

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
