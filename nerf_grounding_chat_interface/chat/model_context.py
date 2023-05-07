import os
import sys
from pathlib import Path
from typing import Optional

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

        print("Initialize visualGrounder")
        visual_grounder = VisualGrounder(
            settings.output_path, settings.camera_poses, lerf_pipeline
        )

        return ModelContext(visual_grounder, blip2captioner, lerf_pipeline)
