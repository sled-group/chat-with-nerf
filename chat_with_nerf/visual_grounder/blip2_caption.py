import torch
from attrs import Factory, define
from PIL import Image

from chat_with_nerf.visual_grounder.image_ref import ImageRef


@define
class Blip2Captioner:
    model: torch.nn.Module
    """BLIP-2 model for captioning."""
    vis_processors: dict
    """Preprocessors for visual inputs.

    txt_processors
    """
    positive_words: str = "computer"

    images: list[ImageRef] = Factory(list)

    def set_positive_words(self, new_positive_words):
        self.positive_words = new_positive_words

    # for the general 6 images pipeline
    def load_images(self, selected: list[ImageRef]) -> None:
        # load sample image
        for image_ref in selected:
            raw_image = Image.open(image_ref.rgb_address).convert("RGB")
            raw_image.resize((596, 437))
            image_ref.raw_image = raw_image
        self.images = selected

    # for the general 6 images pipeline
    def blip2caption(self) -> dict[str, str]:
        """_summary_

        :return: a dictionary of image path and its corresponding caption
        :rtype: dict[str, str]
        """
        # TODO: batch it.
        # prepare the image
        result: dict[str, str] = {}  # key: rgb_address, value: caption
        for image_ref in self.images:
            # path = image_item.getRGBAddress()
            # print("path in blip2caption: ", path)
            image = (
                self.vis_processors["eval"](image_ref.raw_image)
                .unsqueeze(0)
                .to(self.model.device)
            )
            question = (
                "Question: Describe objects focusing on color, size, shape and other"
                " features in the image in detail. Answer:"
                # "Question: Describe the "
                # + self.positive_words
                # + " focusing on color, size, shape and other features in this image?"
                # + "Please answer in the sentence format. Answer:"
            )
            answer = self.model.generate({"image": image, "prompt": question})[0]  # type: ignore
            result[image_ref.rgb_address] = answer

        return result

    def blip2vqa(self, image_path, question):
        raw_image = Image.open(image_path).convert("RGB")
        raw_image.resize((596, 437))
        image = (
            self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.model.device)
        )
        answer = self.model.generate({"image": image, "prompt": question})
        return answer
