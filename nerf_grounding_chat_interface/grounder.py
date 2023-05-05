import logging

import requests

# Set up logging
logging.basicConfig(level=logging.INFO)


def ground(ground_text: str) -> list[tuple[str, str]]:
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
    for img_path, img_caption in data.items():
        # Gradio uses http://localhost:7777/file=/absolute/path/example.jpg to access files,
        # can use relative too, just drop the leading slash
        result.append((f"/file={img_path}", img_caption[0]))

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
