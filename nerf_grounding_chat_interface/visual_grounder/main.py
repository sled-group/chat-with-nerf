import os
import sys
from pathlib import Path
import numpy as np
from nerf_grounding_chat_interface.visual_grounder.setting import Settings
from nerf_grounding_chat_interface.visual_grounder.visual_grounder import VisualGrounder
from nerf_grounding_chat_interface.visual_grounder.blip2_caption import Blip2Captioner
from nerfstudio.pipelines.base_pipeline import Pipeline

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project's root directory to sys.path
sys.path.append(project_root)

print("Get hyperparameters")
setting = Settings()


def initialize():
    load_config = Path(setting.load_config)
    print("load_config: ", setting.load_config)
    output_path = Path(setting.output_path)
    print("output_path: ", setting.output_path)
    print("Initialize visualGrounder")
    visualGrounder = VisualGrounder(load_config, output_path, setting.camera_poses)
    print("Initialize Blip2Captioner")
    # The path of the new working directory
    blip2captioner = Blip2Captioner()
    print("Initialize LERF pipeline")
    pipeline = visualGrounder.construct_pipeline()

    return visualGrounder, blip2captioner, pipeline


def visual_grounder(
    positive_words: str,
    visualGrounder: VisualGrounder,
    blip2captioner: Blip2Captioner,
    pipeline: Pipeline,
):
    # first step: set positive words
    print("Set Positive Words in Visual Grounder")
    print("positive words: ", positive_words)
    visualGrounder.set_positive_words(pipeline, positive_words)
    blip2captioner.set_positive_words(positive_words)
    # second step: take 6 images and enable parallelism
    result, numpy_res = visualGrounder.taking_pictures(pipeline)

    # third step: first select images above the threshold
    selected = []
    for imageItem in result:
        encoding = imageItem.getEncoding()
        # TODO: threshold & histogram
        # False positive
        if np.any(encoding > setting.threshold):
            selected.append(imageItem)

    # fourth step: feed corresponding images to the BLIPv2
    # and acquire descriptions
    blip2captioner.load_images(selected)
    result = blip2captioner.blip2caption()

    # fifth step: send corresponding image and caption pair to the GPT-4
    # return image and comments pair
    return result


def set_positive_words(
    positive_words: str, visualGrounder: VisualGrounder, pipeline: Pipeline
):
    if visualGrounder.set_positive_words(pipeline, positive_words):
        return {"Success": "True"}
    else:
        return {"Success": "False"}


def take_picture(visualGrounder: VisualGrounder, pipeline: Pipeline):
    result = visualGrounder.taking_pictures(pipeline)
    return result
