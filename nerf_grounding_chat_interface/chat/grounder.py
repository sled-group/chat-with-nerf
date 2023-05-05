from nerf_grounding_chat_interface import logger
from nerf_grounding_chat_interface.visual_grounder.main import visual_grounder
from nerf_grounding_chat_interface.chat.config import Config


def ground(
    ground_text: str,
) -> list[tuple[str, str]]:
    # Set the API URL
    logger.info(f"Ground Text: {ground_text}")
    visualGrounder = Config.visualGrounder
    blip2captioner = Config.blip2captioner
    pipeline = Config.pipeline
    response = visual_grounder(ground_text, visualGrounder, blip2captioner, pipeline)
    result = []
    for img_path, img_caption in response:
        # Gradio uses http://localhost:7777/file=/absolute/path/example.jpg to access files,
        # can use relative too, just drop the leading slash
        result.append((f"{img_path}", img_caption[0]))

    return result


def ground_with_callback(
    ground_text,
    callback,
    visual_grounder,
    blip2captioner,
    pipeline,
):
    result = ground(
        ground_text,
        visual_grounder,
        blip2captioner,
        pipeline,
    )
    callback(result)
