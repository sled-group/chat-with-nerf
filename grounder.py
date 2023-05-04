from typing import Tuple, List
from gradio_client import utils as client_utils
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def ground(ground_text: str) -> List[Tuple[str, str]]:
    # Set the API URL
    logging.info(f"Ground Text: {ground_text}")
    url = "http://localhost:7009/visualground/" + ground_text

    # Make an HTTP GET request
    response = requests.get(url)

    # Check the status code
    if response.status_code == 200:
        # Parse the response data (assuming it's JSON)
        data = response.json()

        # Do something with the data, e.g., print it
        logging.info(f"Input value: {data}")
    else:
        logging.info(f"Request failed with status code {response.status_code}")
    
    result = []
    for data_key in data.keys():
        result.append((client_utils.encode_url_or_file_to_base64(data_key), data[data_key]))
        
    return result
    # return [
    #     (
    #         client_utils.encode_url_or_file_to_base64("grounder_images/sofa_1.jpeg"),
    #         "a long sofa with white cover and yellow accent, metallic legs",
    #     ),
    #     (
    #         client_utils.encode_url_or_file_to_base64("grounder_images/sofa_2.jpeg"),
    #         "a loveseat with a pillow on top, white cover and yellow accent, metallic legs",
    #     ),
    # ]

def ground_with_callback(ground_text, callback):
    result = ground(ground_text)
    callback(result)
