import datetime
import json
import os
import uuid
import numpy as np
import cattrs
from attrs import define, field

from chat_with_nerf import logger
from chat_with_nerf.settings import Settings


@define
class Session:
    """A class to store the all information about a session, including the chat
    history and the current scene."""

    session_id: str
    start_time: str
    scene: str
    chat_history_for_llm: list[tuple]
    chat_history_for_display: list[tuple]
    chat_counter: int
    image_id_to_path: dict[int, str] = field(factory=dict)
    grounding_result_mesh_path: str | None = None
    posterior: np.ndarray | None = None

    @classmethod
    def create(cls):
        return Session.create_for_scene(Settings.default_scene)

    @classmethod
    def create_for_scene(cls, scene: str):
        session = cls(
            session_id=str(uuid.uuid4()),
            start_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            scene=scene,
            chat_history_for_llm=[],
            chat_history_for_display=[(None, Settings.INITIAL_MSG_FOR_DISPLAY)],
            chat_counter=0,
        )
        logger.info(
            f"Creating a new session {session.session_id} with scene {session.scene}."
        )
        return session

    def save(self) -> None:
        """Save the session as a json file."""
        logger.info(f"Saving session {self.session_id} to disk.")

        # Create the directory and any parent directories if they don't exist
        os.makedirs(os.path.join(Settings.output_path, self.session_id), exist_ok=True)

        # Write content to the file
        with open(
            os.path.join(Settings.output_path, self.session_id, "session.json"), "w"
        ) as f:
            json.dump(cattrs.unstructure(self), f, indent=4)
        logger.info(f"Session {self.session_id} saved to disk.")
