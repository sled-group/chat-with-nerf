from typing import Any, Optional

import torch
from attrs import Factory, define
from llava.conversation import SeparatorStyle, conv_templates
from llava.model import *  # noqa: F401, F403
from llava.model.utils import KeywordsStoppingCriteria
from PIL import Image
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.image_ref import ImageRef


@define
class BaseCaptioner:
    model: torch.nn.Module
    """Base model for image processing."""
    vis_processors: dict
    """Preprocessors for visual inputs."""

    def process_image(self, image_path: str) -> torch.Tensor:
        """Processes an image and returns it as a tensor."""
        raw_image = Image.open(image_path).convert("RGB")
        return self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.model.device)

    def caption(self):
        """Generates captions for the images."""
        pass


@define
class Blip2Captioner(BaseCaptioner):
    def caption(self, positive_words) -> dict[str, str]:
        """_summary_

        :return: a dictionary of image path and its corresponding caption
        :rtype: dict[str, str]
        """
        # TODO: batch it.
        # prepare the image
        result: dict[str, str] = {}  # key: rgb_address, value: caption
        for image_ref in self.images:
            image = self.process_image(image_ref.raw_image)  # type: ignore
            question = (
                "Describe the shape and material of the"
                + positive_words
                + ", if there is one."
            )
            answer = self.model.generate({"image": image, "prompt": question})[0]  # type: ignore
            result[image_ref.rgb_address] = answer

        return result


@define
class LLaVaCaptioner(BaseCaptioner):
    tokenizer: Optional[Any] = None
    mm_use_im_start_end: bool = True
    image_token_len: int = 512

    def set_positive_words(self, new_positive_words):
        self.positive_words = new_positive_words

    def filter(self, positive_words: str, imagerefs: list[ImageRef]) -> list:
        qs = (
            "Is "
            + positive_words
            + " in this image? Return yes if yes, otherwise return no."
        )
        if self.mm_use_im_start_end:
            qs = (
                qs
                + "\n"
                + Settings.DEFAULT_IM_START_TOKEN
                + Settings.DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
                + Settings.DEFAULT_IM_END_TOKEN
            )
        else:
            qs = qs + "\n" + Settings.DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

        selected = []
        for imageref in imagerefs:
            output = self.llava_output(qs, imageref.rgb_image)
            print("output: ", output)
            if output == "yes":
                selected.append(imageref)
        return selected

    def caption(self, positive_words: str, imagerefs: list[ImageRef]) -> dict[str, str]:
        """_summary_

        :return: a dictionary of image path and its corresponding caption
        :rtype: dict[str, str]
        """
        qs = (
            "Is there a "
            + positive_words
            + " in the image? If there is, "
            + "describe it "
            + "in detail, like its size, color, shape, material and relations to its"
            + " surrounding objects."
        )

        print(qs)

        if self.mm_use_im_start_end:
            qs = (
                qs
                + "\n"
                + Settings.DEFAULT_IM_START_TOKEN
                + Settings.DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
                + Settings.DEFAULT_IM_END_TOKEN
            )
        else:
            qs = qs + "\n" + Settings.DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

        result: dict[str, str] = {}  # key: rgb_address, value: caption
        for image_ref in imagerefs:
            image = image_ref.rgb_image
            outputs = self.llava_output(qs, image)
            result[image_ref.rgb_address] = outputs
        return result

    def llava_output(self, qs: str, image: Image):
        conv_mode = "multimodal"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])  # type: ignore

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        input_token_len = input_ids.shape[1]
        image_tensor = self.vis_processors["image_processor"].preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
            )  # type: ignore
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids "
                f"are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(  # type: ignore
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[
            0
        ]  # type: ignore
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs
