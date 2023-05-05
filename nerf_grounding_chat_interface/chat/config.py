from attrs import define
from nerf_grounding_chat_interface.visual_grounder.blip2_caption import Blip2Captioner
from nerf_grounding_chat_interface.visual_grounder.visual_grounder import VisualGrounder
from nerfstudio.pipelines.base_pipeline import Pipeline


@define
class Config:
    visualGrounder: VisualGrounder
    blip2captioner: Blip2Captioner
    pipeline: Pipeline
