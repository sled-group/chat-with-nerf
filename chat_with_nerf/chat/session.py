import uuid

from attrs import define

from chat_with_nerf import logger
from chat_with_nerf.settings import Settings


@define
class Session:
    """A class to store the all information about a session, including the chat
    history and the current scene."""

    session_id: str
    scene: str
    chat_history_for_llm: list[tuple]
    chat_history_for_display: list[tuple]
    chat_counter: int

    @classmethod
    def create(cls):
        return Session.create_for_scene(Settings.default_scene)

    @classmethod
    def create_for_scene(cls, scene: str):
        session = cls(
            session_id=str(uuid.uuid4()),
            scene=scene,
            chat_history_for_llm=[],
            chat_history_for_display=[(None, Settings.INITIAL_MSG_FOR_DISPLAY)],
            chat_counter=0,
        )
        logger.info(
            f"Creating a new session {session.session_id} with scene {session.scene}."
        )
        return session

    def save(self):
        # TODO: save to disk/db
        pass
