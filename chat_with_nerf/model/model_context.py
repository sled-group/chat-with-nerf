import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from attrs import Factory, define

# from lavis.models import load_model_and_preprocess  # type: ignore
from llava import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
)

from chat_with_nerf import logger
from chat_with_nerf.model.scene_config import SceneConfig
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.captioner import (  # Blip2Captioner,
    BaseCaptioner,
    LLaVaCaptioner,
)
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


@define
class ModelContext:
    scene_configs: dict[str, SceneConfig] | None
    visual_grounder: dict[str, VisualGrounder] | None
    pipeline: dict[str, Pipeline] | None
    captioner: BaseCaptioner
    clip_model: CLIPModel
    clip_processor: CLIPProcessor


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

        # logger.info("Search for all Scenes and Set the current Scene")
        # scene_configs = ModelContextManager.search_scenes(Settings.data_path)

        logger.info("Initialize Captioner")
        if Settings.TYPE_CAPTIONER == "blip2":
            print("blip2")
            # captioner = ModelContextManager.initiaze_blip_captioner()
        else:
            captioner = ModelContextManager.initiaze_llava_captioner()

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return ModelContext(None, None, None, captioner, clip_model, clip_processor)

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

    # @staticmethod
    # def initiaze_blip_captioner() -> Blip2Captioner:
    #     model, vis_processors, _ = load_model_and_preprocess(
    #         name="blip2_t5",
    #         model_type="pretrain_flant5xxl",
    #         is_eval=True,
    #         device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    #     )

    #     return Blip2Captioner(model, vis_processors)

    @staticmethod
    def initiaze_llava_captioner() -> LLaVaCaptioner:
        disable_torch_init()
        model_name = os.path.expanduser(Settings.LLAVA_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
        ).cuda()
        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )

        image_processor = {"image_processor": image_processor}
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([Settings.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [Settings.DEFAULT_IM_START_TOKEN, Settings.DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
        vision_tower = model.get_model().vision_tower[0]
        if vision_tower.device.type == "meta":
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower.config._name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).cuda()
            model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device="cuda", dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [Settings.DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            (
                vision_config.im_start_token,
                vision_config.im_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [Settings.DEFAULT_IM_START_TOKEN, Settings.DEFAULT_IM_END_TOKEN]
            )
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        captioner = LLaVaCaptioner(
            model,
            image_processor,
            tokenizer,
            mm_use_im_start_end,
            image_token_len,
        )
        return captioner

    @staticmethod
    def initialize_lerf_pipeline(load_config: str) -> Pipeline:
        _, lerf_pipeline, _, _ = eval_setup(
            Path(load_config),
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )

        return lerf_pipeline
