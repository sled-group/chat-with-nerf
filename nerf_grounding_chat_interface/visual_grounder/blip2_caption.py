import torch
from attrs import Factory, define
from lavis.models import load_model_and_preprocess  # type: ignore
from PIL import Image

from nerf_grounding_chat_interface.visual_grounder.image_ref import ImageRef


@define
class Blip2Captioner:
    positive_words: str = "computer"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xxl",
        is_eval=True,
        device=device,
    )
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
                .to(self.device)
            )
            question = (
                "Question: Describe the "
                + self.positive_words
                + " focusing on color, size, shape and other features in this image?"
                + "Please answer in the sentence format. Answer:"
            )
            answer = self.model.generate({"image": image, "prompt": question})[0]
            result[image_ref.rgb_address] = answer

        return result

    def blip2vqa(self, image_path, question):
        raw_image = Image.open(image_path).convert("RGB")
        raw_image.resize((596, 437))
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        answer = self.model.generate({"image": image, "prompt": question})
        return answer
