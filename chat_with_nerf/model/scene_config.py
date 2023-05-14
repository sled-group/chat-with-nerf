from attrs import define


@define
class SceneConfig:
    load_lerf_config: str
    camera_path: str
    camera_poses: dict
