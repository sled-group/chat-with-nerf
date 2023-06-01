import pytest

from chat_with_nerf.chat.grounder import ground


# Pytest fixtures for the ground method arguments
@pytest.fixture
def session_id():
    return "session_id"


@pytest.fixture
def dropdown_scene():
    return "test_scene"


@pytest.fixture
def ground_text():
    return "example text"


@pytest.fixture
def visual_grounder(mocker):
    return mocker.Mock()


@pytest.fixture
def captioner(mocker):
    return mocker.Mock()


@pytest.fixture
def session(mocker):
    return mocker.Mock()


@pytest.fixture
def picture_taker(mocker):
    return mocker.Mock()


# Test success case
def test_ground_success(
    mocker,
    dropdown_scene,
    ground_text,
    picture_taker,
    captioner,
    session,
):
    mocker.patch(
        "chat_with_nerf.chat.grounder.VisualGrounder.call_visual_grounder",
        return_value=(
            {"/path/to/image.jpg": "This is a caption."},
            "some/path/to/mesh.glb",
        ),
    )

    result = ground(session, dropdown_scene, ground_text, picture_taker, captioner)

    assert result == [("/path/to/image.jpg", "This is a caption.")]


# Test failure case (e.g., when call_visual_grounder raises an exception)
def test_ground_failure(
    mocker,
    dropdown_scene,
    ground_text,
    captioner,
    session,
):
    mocker.patch(
        "chat_with_nerf.chat.grounder.VisualGrounder.call_visual_grounder",
        side_effect=Exception("Something went wrong."),
    )

    with pytest.raises(Exception, match="Something went wrong."):
        ground(session, dropdown_scene, ground_text, picture_taker, captioner)
