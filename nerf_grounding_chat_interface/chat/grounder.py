from typing import Callable

from nerfstudio.pipelines.base_pipeline import Pipeline

from nerf_grounding_chat_interface import logger
from nerf_grounding_chat_interface.visual_grounder.blip2_caption import Blip2Captioner
from nerf_grounding_chat_interface.visual_grounder.main import call_visual_grounder
from nerf_grounding_chat_interface.visual_grounder.visual_grounder import VisualGrounder


def ground(
    ground_text: str,
    visual_grounder: VisualGrounder,
    blip2captioner: Blip2Captioner,
    pipeline: Pipeline,
) -> list[tuple[str, str]]:
    """Ground a text in a scene by returning the relavant images and their
    corresponding captions.

    :param ground_text: the text query to be grounded
    :type ground_text: str
    :param visual_grounder: a visual grounder model
    :type visual_grounder: VisualGrounder
    :param blip2captioner: a blip2captioner model
    :type blip2captioner: Blip2Captioner
    :param pipeline: a pipeline model
    :type pipeline: Pipeline
    """

    logger.info(f"Ground Text: {ground_text}")

    response = call_visual_grounder(
        ground_text, visual_grounder, blip2captioner, pipeline
    )
    result = []
    for img_path, img_caption in response.items():
        # Gradio uses http://localhost:7777/file=/absolute/path/example.jpg to access files,
        # can use relative too, just drop the leading slash
        result.append((img_path, img_caption))

    return result


def ground_with_callback(
    ground_text: str,
    visual_grounder: VisualGrounder,
    blip2captioner: Blip2Captioner,
    pipeline: Pipeline,
    callback: Callable[[list[tuple[str, str]]], None],
):
    result = ground(
        ground_text,
        visual_grounder,
        blip2captioner,
        pipeline,
    )
    callback(result)
