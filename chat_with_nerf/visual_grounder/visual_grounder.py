from __future__ import annotations

import numpy as np
import torch
import clip
from attrs import define
from rich.console import Console
from chat_with_nerf.chat.session import Session
from chat_with_nerf import logger
from chat_with_nerf.visual_grounder.captioner import BaseCaptioner
from chat_with_nerf.visual_grounder.picture_taker import PictureTaker
from chat_with_nerf.settings import Settings

CONSOLE = Console(width=120)


@define
class VisualGrounder:
    @staticmethod
    def call_visual_grounder(
        session: Session,
        positive_words: str,
        picture_taker: PictureTaker,
        captioner: BaseCaptioner,
    ) -> tuple[dict[str, str] | None, str | None]:
        """Return a dictionary of image path and its corresponding caption.

        :return: a dictionary of image path and its corresponding caption and the mesh path
        """

        # first step: set positive words
        logger.debug("Set Positive Words in Visual Grounder")
        logger.debug("positive words: " + positive_words)

        image_refs, grounding_result_mesh_path = picture_taker.take_picture(
            positive_words, session
        )

        logger.info(f"Took {len(image_refs)} pictures.")

        if len(image_refs) == 0:
            captioner_result = None
        else:
            captioner_result = captioner.caption(positive_words, image_refs)

        torch.cuda.empty_cache()  # free up GPU memory

        return captioner_result, grounding_result_mesh_path

    @staticmethod
    def call_visual_grounder_no_gpt(
        session: Session,
        positive_words: str,
        picture_taker: PictureTaker,
    ):
        logger.debug("Set Positive Words in Visual Grounder")
        logger.debug("positive words: " + positive_words)

        (
            center_list,
            box_size_list,
            values_list,
        ) = picture_taker.visual_ground_pipeline_no_gpt(
            positive_words, session.session_id
        )
        return center_list, box_size_list, values_list

    @staticmethod
    def target_finder(
        session: Session,
        positive_phrase: str,
        picture_taker: PictureTaker,
    ):
        (
            centroids,
            bboxes,
        ), paths2images = picture_taker.visual_ground_pipeline_with_gpt(
            positive_phrase, session
        )

        return (centroids, bboxes), paths2images

    @staticmethod
    def landmark_finder(
        session: Session,
        positive_phrase: str,
        picture_taker: PictureTaker,
    ):
        centroids = picture_taker.visual_ground_pipeline_with_gpt_lerf(
            positive_phrase, session.session_id
        )

        return centroids

    @staticmethod
    def visual_feedback(positive_phrase, target_candidate_images_list, picture_taker):
        clip_model = picture_taker.clip_model
        clip_tokenizer = picture_taker.clip_tokenizer
        images = [img_ref.raw_image for img_ref in target_candidate_images_list]

        image_input = torch.tensor(np.stack(images)).cuda()
        natural_sentences = [positive_phrase]
        # print(image_input.shape) torch.Size([21, 3, 224, 224])
        text_tokens = clip_tokenizer.tokenize(natural_sentences).cuda()
        with torch.no_grad():
            clip_text_features = clip_model.encode_text(text_tokens).float()
            clip_image_features = clip_model.encode_image(image_input).float()
        clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)

        similarity = (
            clip_text_features.cpu().numpy() @ clip_image_features.cpu().numpy().T
        )

        return similarity.tolist()

    @staticmethod
    def target_finder_openscene(
        session: Session,
        positive_phrase: str,
        picture_taker: PictureTaker,
    ):
        (
            centroids,
            bboxes,
            similarity_mean_list,
        ) = picture_taker.visual_ground_target_finder_with_gpt_openscene(
            positive_phrase, session.session_id
        )

        return (centroids, bboxes)

    @staticmethod
    def landmark_finder_openscene(
        session: Session,
        positive_phrase: str,
        picture_taker: PictureTaker,
    ):
        (
            centroid,
            extend,
        ) = picture_taker.visual_ground_landmark_finder_with_gpt_openscene(
            positive_phrase, session.session_id
        )

        return centroid, extend

    @staticmethod
    def visual_feedback_openscene(
        positive_phrase, target_candidate_images_list, picture_taker
    ):
        ## TODO: take a look.
        clip_model = picture_taker.clip_model
        images = [
            picture_taker.clip_preprocess(img_ref.raw_image)
            for img_ref in target_candidate_images_list
        ]

        image_input = torch.tensor(np.stack(images)).cuda()
        natural_sentences = [positive_phrase]
        # print(image_input.shape) torch.Size([21, 3, 224, 224])
        text_tokens = clip.tokenize(natural_sentences).cuda()
        with torch.no_grad():
            clip_text_features = clip_model.encode_text(text_tokens).float()
            clip_image_features = clip_model.encode_image(image_input).float()
        clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
        # linear probing maybe worth a shot
        similarity = (
            clip_text_features.cpu().numpy() @ clip_image_features.cpu().numpy().T
        )

        return similarity.tolist()
