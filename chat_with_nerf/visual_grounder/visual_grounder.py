from __future__ import annotations

from attrs import define
from rich.console import Console

from chat_with_nerf import logger
from chat_with_nerf.visual_grounder.captioner import BaseCaptioner
from chat_with_nerf.visual_grounder.picture_taker import PictureTaker

CONSOLE = Console(width=120)


@define
class VisualGrounder:
    @staticmethod
    def call_visual_grounder(
        session_id: str,
        positive_words: str,
        picture_taker: PictureTaker,
        captioner: BaseCaptioner,
    ) -> dict[str, str]:
        """Return a dictionary of image path and its corresponding caption.

        :return: a dictionary of image path and its corresponding caption
        """

        # first step: set positive words
        logger.debug("Set Positive Words in Visual Grounder")
        logger.debug("positive words: ", positive_words)

        image_refs = picture_taker.take_picture(positive_words, session_id)

        # captioner.load_images(image_refs)
        captioner_result = captioner.caption(positive_words, image_refs)

        return captioner_result
