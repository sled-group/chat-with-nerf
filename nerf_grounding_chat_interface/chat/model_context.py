from attrs import define
from nerfstudio.pipelines.base_pipeline import Pipeline

from nerf_grounding_chat_interface.visual_grounder.blip2_caption import Blip2Captioner
from nerf_grounding_chat_interface.visual_grounder.visual_grounder import VisualGrounder


@define
class ModelContext:
    visualGrounder: VisualGrounder
    blip2captioner: Blip2Captioner
    pipeline: Pipeline
