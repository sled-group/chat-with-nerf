from attrs import define


@define
class SceneConfig:
    scene_name: str
    load_lerf_config: str
    load_h5_config: str
    camera_path: str
    nerf_exported_mesh_path: str
    load_openscene: str
    load_mesh: str
    load_metadata: str
