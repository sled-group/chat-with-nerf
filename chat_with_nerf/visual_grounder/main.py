import glob
import os
from pathlib import Path

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from chat_with_nerf import logger
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.captioner import BaseCaptioner
from chat_with_nerf.visual_grounder.image_ref import ImageRef
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


def call_visual_grounder(
    session_id: str,
    dropdown_scene: str,
    positive_words: str,
    visual_grounder: VisualGrounder,
    captioner: BaseCaptioner,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> dict[str, str]:
    """Return a dictionary of image path and its corresponding caption.

    :return: a dictionary of image path and its corresponding caption
    """

    # first step: set positive words
    logger.debug("Set Positive Words in Visual Grounder")
    logger.debug("positive words: " + positive_words)
    # visual_grounder.set_positive_words(positive_words)
    # second step: take 6 images and enable parallelism
    # grounder_result = visual_grounder.taking_pictures(session_id)

    # third step: first select images above the threshold
    imagerefs = load_images(dropdown_scene)
    selected = filter(imagerefs, clip_model, clip_processor, positive_words)
    # selected = captioner.filter(positive_words, selected)
    # fourth step: feed corresponding images to the BLIPv2
    # and acquire descriptions
    # captioner.load_images(selected)
    captioner_result = captioner.caption(positive_words, selected)

    # fifth step: send corresponding image and caption pair to the GPT-4
    # return image and comments pair
    return captioner_result


def load_images(dropdown_scene) -> list[ImageRef]:
    # specify your path
    path = Path(Settings.IMAGES_PATH) / Path(dropdown_scene)

    # use glob to get all the png images
    image_files = glob.glob(os.path.join(path, "*.png"))

    # create a list to hold the images
    imagerefs = []

    # open each image and append it to the images list
    for image_file in image_files:
        image = Image.open(image_file)
        imageref = ImageRef(str(path / Path(image_file)), image)
        imagerefs.append(imageref)

    return imagerefs


def filter(
    imagerefs: list[ImageRef],
    model: CLIPModel,
    processor: CLIPProcessor,
    positive_words: str,
) -> list:
    top = None
    top_clip_score = 0
    selected = []
    for imageref in imagerefs:
        inputs = processor(
            text=[positive_words],
            images=imageref.rgb_image,
            return_tensors="pt",
            padding=True,
        )

        outputs = model(**inputs)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        if logits_per_image > top_clip_score:
            top_clip_score = logits_per_image
            top = imageref
        print("logits_per_image: ", logits_per_image.item())
        if logits_per_image.item() > Settings.CLIP_FILTERING_THRESHOLD:
            selected.append(imageref)

    return selected if len(selected) > 0 else [top]


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
