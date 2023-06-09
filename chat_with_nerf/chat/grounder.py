from typing import Callable

from chat_with_nerf import logger
from chat_with_nerf.chat.session import Session
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.captioner import BaseCaptioner
from chat_with_nerf.visual_grounder.picture_taker import PictureTaker
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


def ground(
    session: Session,
    dropdown_scene: str,
    ground_text: str,
    picture_taker: PictureTaker,
    captioner: BaseCaptioner,
) -> list[tuple[str, str]] | None:
    """Ground a text in a scene by returning the relavant images and their
    corresponding captions.

    :param ground_text: the text query to be grounded
    :type ground_text: str
    :param visual_grounder: a visual grounder model
    :type visual_grounder: VisualGrounder
    :param captioner: a BaseCaptioner model
    :type captioner: BaseCaptioner
    """

    if Settings.USE_FAKE_GROUNDER:
        print("FAKE: ", Settings.USE_FAKE_GROUNDER)
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
    # TODO: fix this!
    captioner_result, grounding_result_mesh_path = VisualGrounder.call_visual_grounder(
        session.session_id, ground_text, picture_taker, captioner
    )
    logger.debug(f"call_visual_grounder captioner_result: {captioner_result}")
    logger.debug(
        f"call_visual_grounder grounding_result_mesh_path: {grounding_result_mesh_path}"
    )

    session.grounding_result_mesh_path = grounding_result_mesh_path

    if captioner_result is None:
        return None

    result = []
    for img_path, img_caption in captioner_result.items():
        # Gradio uses http://localhost:7777/file=/absolute/path/example.jpg to access files,
        # can use relative too, just drop the leading slash
        result.append((img_path, img_caption))

    return result


def ground_with_callback(
    session: Session,
    dropdown_scene: str,
    ground_text: str,
    picture_taker: PictureTaker,
    captioner: BaseCaptioner,
    callback: Callable[[list[tuple[str, str]] | None, Session], None],
):
    result = ground(
        session,
        dropdown_scene,
        ground_text,
        picture_taker,
        captioner,
    )
    callback(result, session)
