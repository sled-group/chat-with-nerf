import os
import sys

import numpy as np
from nerfstudio.pipelines.base_pipeline import Pipeline

from nerf_grounding_chat_interface.visual_grounder.blip2_caption import Blip2Captioner
from nerf_grounding_chat_interface.visual_grounder.image_ref import ImageRef
from nerf_grounding_chat_interface.visual_grounder.setting import Settings
from nerf_grounding_chat_interface.visual_grounder.visual_grounder import VisualGrounder

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project's root directory to sys.path
sys.path.append(project_root)

print("Get hyperparameters")
setting = Settings()


def call_visual_grounder(
    positive_words: str,
    visual_grounder: VisualGrounder,
    blip2captioner: Blip2Captioner,
    pipeline: Pipeline,
) -> dict[str, str]:
    """Return a dictionary of image path and its corresponding caption.

    :return: a dictionary of image path and its corresponding caption
    """

    # first step: set positive words
    print("Set Positive Words in Visual Grounder")
    print("positive words: ", positive_words)
    visual_grounder.set_positive_words(pipeline, positive_words)
    blip2captioner.set_positive_words(positive_words)
    # second step: take 6 images and enable parallelism
    grounder_result = visual_grounder.taking_pictures(pipeline)

    # third step: first select images above the threshold
    selected: list[ImageRef] = []
    for imageItem in grounder_result:
        encoding = imageItem.encoding
        # TODO: threshold & histogram
        # False positive
        if np.any(encoding > setting.threshold):
            selected.append(imageItem)

    # fourth step: feed corresponding images to the BLIPv2
    # and acquire descriptions
    blip2captioner.load_images(selected)
    captioner_result = blip2captioner.blip2caption()

    # fifth step: send corresponding image and caption pair to the GPT-4
    # return image and comments pair
    return captioner_result


def set_positive_words(
    positive_words: str, visualGrounder: VisualGrounder, pipeline: Pipeline
):
    if visualGrounder.set_positive_words(pipeline, positive_words):
        return {"Success": "True"}
    else:
        return {"Success": "False"}


def take_picture(visual_grounder: VisualGrounder, pipeline: Pipeline):
    result = visual_grounder.taking_pictures(pipeline)
    return result
