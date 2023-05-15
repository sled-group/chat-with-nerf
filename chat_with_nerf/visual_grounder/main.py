import numpy as np

from chat_with_nerf import logger
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.blip2_caption import Blip2Captioner
from chat_with_nerf.visual_grounder.image_ref import ImageRef
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


def call_visual_grounder(
    session_id: str,
    positive_words: str,
    visual_grounder: VisualGrounder,
    blip2captioner: Blip2Captioner,
) -> dict[str, str]:
    """Return a dictionary of image path and its corresponding caption.

    :return: a dictionary of image path and its corresponding caption
    """

    # first step: set positive words
    logger.debug("Set Positive Words in Visual Grounder")
    logger.debug("positive words: ", positive_words)
    visual_grounder.set_positive_words(positive_words)
    blip2captioner.set_positive_words(positive_words)
    # second step: take 6 images and enable parallelism
    grounder_result = visual_grounder.taking_pictures(session_id)

    # third step: first select images above the threshold
    selected: list[ImageRef] = []
    for imageItem in grounder_result:
        encoding = imageItem.encoding
        # TODO: threshold & histogram
        # False positive
        if np.any(encoding > Settings().threshold):
            selected.append(imageItem)

    # fourth step: feed corresponding images to the BLIPv2
    # and acquire descriptions
    blip2captioner.load_images(selected)
    captioner_result = blip2captioner.blip2caption()

    # fifth step: send corresponding image and caption pair to the GPT-4
    # return image and comments pair
    return captioner_result


def set_positive_words(
    positive_words: str,
    visualGrounder: VisualGrounder,
):
    if visualGrounder.set_positive_words(positive_words):
        return {"Success": "True"}
    else:
        return {"Success": "False"}


def take_picture(visual_grounder: VisualGrounder, session_id: str):
    result = visual_grounder.taking_pictures(session_id)
    return result
