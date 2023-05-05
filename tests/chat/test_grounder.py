import pytest

from nerf_grounding_chat_interface.chat.grounder import ground


@pytest.mark.parametrize("status_code", [500, 404, 400])
def test_ground_request_failure(mocker, status_code):
    # Mock the requests.get function
    mock_get = mocker.patch("requests.get")

    # Create a failed response object with a status code of 400
    mock_response = mocker.Mock()
    mock_response.status_code = status_code

    # Configure the mock get function to return the mock response
    mock_get.return_value = mock_response

    # Call the ground function and check if it raises the expected exception
    with pytest.raises(
        ValueError, match="Request failed with status code " + str(status_code)
    ):
        ground("sample_text")


def test_ground_success(mocker):
    # Mock the requests.get function
    mock_get = mocker.patch("requests.get")

    # Create a successful response object with a status code of 200 and JSON data
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "/img1.jpg": ["caption1"],
        "/img2.jpg": ["caption2"],
    }

    # Configure the mock get function to return the mock response
    mock_get.return_value = mock_response

    # Call the ground function and check the result
    result = ground("sample_text")
    assert result == [("/img1.jpg", "caption1"), ("/img2.jpg", "caption2")]
