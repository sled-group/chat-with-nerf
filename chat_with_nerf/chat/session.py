import datetime
import json
import os
import uuid
import numpy as np
import cattrs
from attrs import define, field
import pickle

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
    ground_result: list[tuple[float]] | None = None
    candidate: list | None = None
    chosen_candidate_id: int | None = None
    working_scene_name: str | None = None
    grounding_query: str | None = None
    ground_truth: list | None = None
    top_5_objects2scores: dict | None = None
    center_list: list | None = None
    box_size_list: list | None = None
    values_list: list | None = None
    base_mesh_path: str | None = None
    candidate_visualization: list | None = None
    landmark_visualization: list | None = None
    camera_poses: list | None = None

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
        session.working_scene_name = scene
        return session

    def convert_float32(self, obj):
        """Convert all np.float32 values in the given object to Python float."""
        if isinstance(obj, np.float32):
            return float(obj)

        if isinstance(obj, list):
            return [self.convert_float32(item) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self.convert_float32(item) for item in obj)

        if isinstance(obj, dict):
            return {key: self.convert_float32(value) for key, value in obj.items()}

        return obj

    def save(self, output_path: str) -> None:
        """Save the session as a json file."""
        logger.info(f"Saving session {self.session_id} to disk.")

        # Create the directory and any parent directories if they don't exist
        os.makedirs(os.path.join(output_path, self.working_scene_name), exist_ok=True)

        structured_data = cattrs.unstructure(self)
        structured_data.pop("chat_history_for_display", None)
        # Convert all np.float32 to float
        converted_data = self.convert_float32(structured_data)

        with open(
            os.path.join(
                output_path, self.working_scene_name, f"{self.session_id}.json"
            ),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(converted_data, file, indent=4)
        logger.info(f"Session {self.session_id} saved to disk.")
