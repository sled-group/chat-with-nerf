class Settings:
    data_path: str = "/workspace/chat-with-nerf/data"
    output_path: str = "/workspace/chat-with-nerf/session_output"
    threshold: float = 0.7
    default_scene: str = "office"
    INITIAL_MSG_FOR_DISPLAY = "Hello there! What can I help you find in this room?"
    USE_FAKE_GROUNDER: bool = False
