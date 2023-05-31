import numpy as np
from attrs import define


@define
class CameraPose:
    camera_type: str = "perspective"
    render_height: int = 512
    render_width: int = 512

    def construct_camera_pose(self, c2w: np.ndarray) -> dict:
        camera_pose: dict[str, int | str | list[dict] | None] = {}
        camera_pose["camera_type"] = self.camera_type
        camera_pose["render_height"] = self.render_height
        camera_pose["render_width"] = self.render_width
        camera_pose["camera_path"] = []

        # for c2w in c2ws:
        c2w_dict: dict[str, np.ndarray | int] = {}
        c2w_dict["camera_to_world"] = c2w
        c2w_dict["fov"] = 60
        c2w_dict["aspect"] = 1
        camera_pose["camera_path"].append(c2w_dict)  # type: ignore

        camera_pose["crop"] = None

        return camera_pose
