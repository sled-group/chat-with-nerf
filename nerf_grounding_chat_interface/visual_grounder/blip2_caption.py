import torch
from PIL import Image
from lavis.models import load_model_and_preprocess  # type: ignore
from attrs import define


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

    def set_positive_words(self, new_positive_words):
        self.positive_words = new_positive_words

    # for the general 6 images pipeline
    def load_images(self, selected):
        # load sample image
        self.images = {}
        for selected_item in selected:
            raw_image = Image.open(selected_item.getRGBAddress()).convert("RGB")
            raw_image.resize((596, 437))
            self.images[selected_item] = raw_image

    # for the general 6 images pipeline
    def blip2caption(self):
        # TODO: batch it.
        # prepare the image
        result = {}
        for image_item in self.images.keys():
            # path = image_item.getRGBAddress()
            # print("path in blip2caption: ", path)
            image = (
                self.vis_processors["eval"](self.images[image_item])
                .unsqueeze(0)
                .to(self.device)
            )
            question = (
                "Question: Describe the "
                + self.positive_words
                + " focusing on color, size, shape and other features in this image?"
                + "Please answer in the sentence format. Answer:"
            )
            answer = self.model.generate({"image": image, "prompt": question})
            result[image_item.getRGBAddress()] = answer

        return result

    def blip2vqa(self, image_path, question):
        raw_image = Image.open(image_path).convert("RGB")
        raw_image.resize((596, 437))
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        answer = self.model.generate({"image": image, "prompt": question})
        return answer
