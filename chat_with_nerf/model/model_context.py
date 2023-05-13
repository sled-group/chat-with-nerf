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

from chat_with_nerf.model.scene import Scene
from chat_with_nerf.settings import Settings  # type: ignore
from chat_with_nerf.visual_grounder.blip2_caption import Blip2Captioner
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


@define
class ModelContext:
    visual_grounder: dict[str, VisualGrounder]
    scene_configs: dict[str, Scene]
    blip2captioner: Blip2Captioner
    pipeline: dict[str, Pipeline]


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

        settings = Settings()

        print("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(settings.data_path)

        print("Initialize Blip2Captioner")
        blip2captioner = ModelContextManager.initiaze_blip_captioner()

        print("Initialize LERF pipelines and visualGrounder for all scenes")
        pipeline = {}
        visual_grounder_ins = {}
        initial_dir = os.getcwd()
        for scene_name, scene in scene_configs.items():
            # LERF's implementation requires to find output directory
            os.chdir(settings.data_path + "/" + scene_name)
            lerf_pipeline = ModelContextManager.initialize_lerf_pipeline(
                scene.load_lerf_config
            )
            pipeline[scene_name] = lerf_pipeline
            visual_grounder_ins[scene_name] = VisualGrounder(
                settings.output_path, scene.camera_poses, lerf_pipeline
            )

        # move back the current directory
        os.chdir(initial_dir)
        return ModelContext(
            visual_grounder_ins, scene_configs, blip2captioner, pipeline
        )

    @staticmethod
    def search_scenes(path: str) -> dict[str, Scene]:
        scenes = {}
        subdirectories = [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]

        for subdirectory in subdirectories:
            scene_path = (Path(path) / subdirectory / subdirectory).with_suffix(".yaml")
            with open(scene_path) as f:
                data = yaml.safe_load(f)
            scene = Scene(
                data["load_lerf_config"], data["camera_path"], data["camera_poses"]
            )
            scenes[subdirectory] = scene
        return scenes

    @staticmethod
    def get_current_scene_config(
        scene_name: str, scene_configs: dict[str, Scene]
    ) -> tuple[str, str]:
        return (
            scene_configs[scene_name].camera_path,
            scene_configs[scene_name].load_lerf_config,
        )

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
