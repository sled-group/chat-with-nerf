import os
import sys
from pathlib import Path
from typing import Optional

from nerf_grounding_chat_interface.chat.model_context import ModelContext
from nerf_grounding_chat_interface.visual_grounder.blip2_caption import Blip2Captioner
from nerf_grounding_chat_interface.visual_grounder.setting import Settings
from nerf_grounding_chat_interface.visual_grounder.visual_grounder import VisualGrounder

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project's root directory to sys.path
sys.path.append(project_root)

print("Get hyperparameters")
setting = Settings()

model_context: Optional[ModelContext] = None


# initialize the blipv2 and LERF
def initialize_model_context() -> ModelContext:
    load_config = Path(setting.load_config)
    print("load_config: ", setting.load_config)
    output_path = Path(setting.output_path)
    print("output_path: ", setting.output_path)
    print("Initialize visualGrounder")
    visual_grounder = VisualGrounder(load_config, output_path, setting.camera_poses)
    print("Initialize Blip2Captioner")
    # The path of the new working directory
    blip2captioner = Blip2Captioner()
    print("Initialize LERF pipeline")
    pipeline = visual_grounder.construct_pipeline()

    return ModelContext(visual_grounder, blip2captioner, pipeline)


def get_model_context():
    global model_context
    if model_context is None:
        model_context = initialize_model_context()
    return model_context
