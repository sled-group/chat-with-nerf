from typing import Callable

from chat_with_nerf import logger
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.blip2_caption import Blip2Captioner
from chat_with_nerf.visual_grounder.main import call_visual_grounder
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


def ground(
    ground_text: str,
    visual_grounder: VisualGrounder,
    blip2captioner: Blip2Captioner,
) -> list[tuple[str, str]]:
    """Ground a text in a scene by returning the relavant images and their
    corresponding captions.

    :param ground_text: the text query to be grounded
    :type ground_text: str
    :param visual_grounder: a visual grounder model
    :type visual_grounder: VisualGrounder
    :param blip2captioner: a blip2captioner model
    :type blip2captioner: Blip2Captioner
    """

    print("FAKE: ", Settings.USE_FAKE_GROUNDER)
    settings = Settings()
    if settings.USE_FAKE_GROUNDER:
        return [
            (
                "/workspace/chat-with-nerf/grounder_output/rgb/000.png",
                "a long sofa with white cover and yellow accent, metallic legs",
            ),
            (
                "/workspace/chat-with-nerf/grounder_output/rgb/001.png",
                "a loveseat with a pillow on top, white cover and yellow accent, metallic legs",
            ),
        ]

    logger.info(f"Ground Text: {ground_text}")

    response = call_visual_grounder(ground_text, visual_grounder, blip2captioner)
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
    callback: Callable[[list[tuple[str, str]]], None],
):
    result = ground(
        ground_text,
        visual_grounder,
        blip2captioner,
    )
    callback(result)
