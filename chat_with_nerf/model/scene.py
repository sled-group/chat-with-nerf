import attr


@attr.s
class Scene:
    load_lerf_config: str = attr.ib()
    camera_path: str = attr.ib()
    camera_poses: dict = attr.ib()
