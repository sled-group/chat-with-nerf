from typing import Tuple, List
from gradio_client import utils as client_utils


def ground(ground_text: str) -> List[Tuple[str, str]]:
    return [
        (
            client_utils.encode_url_or_file_to_base64("grounder_images/sofa_1.jpeg"),
            "a long sofa with white cover and yellow accent, metallic legs",
        ),
        (
            client_utils.encode_url_or_file_to_base64("grounder_images/sofa_2.jpeg"),
            "a loveseat with a pillow on top, white cover and yellow accent, metallic legs",
        ),
    ]
