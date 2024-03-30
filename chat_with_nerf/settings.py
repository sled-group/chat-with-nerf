class Chat_With_NeRF_Settings:
    data_path: str = "/workspace/chat-with-nerf-dev/chat-with-nerf/data"
    output_path: str = "/workspace/chat-with-nerf-dev/chat-with-nerf/session_output"
    CLIP_FILTERING_THRESHOLD: float = 21  # range is (0, 100)
    default_scene: str = "scene0025_00"
    INITIAL_MSG_FOR_DISPLAY = "Hello there! What can I help you find in this room?"
    MAX_TURNS = 10
    USE_FAKE_GROUNDER: bool = False
    TYPE_CAPTIONER = "llava"
    LLAVA_PATH = "/workspace/chat-with-nerf-dev/pre-trained-weights/LLaVA/LLaVA-13B-v0"
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    MAX_WORKERS = 5
    IMAGES_PATH = "/workspace/chat-with-nerf-dev/chat-with-nerf/scene_images"
    NERF_DATA_PATH = "/workspace/chat-with-nerf-dev/chat-with-nerf/data"
    NO_GPT: bool = False
    TOP_THREE_NO_GPT: bool = False
    USE_OPENSCENE: bool = True
    # this flag is reserved for in the wild data
    IS_SCANNET: bool = False
    # this flag is only used for evaluation
    IS_EVALUATION: bool = False


Settings = Chat_With_NeRF_Settings()
