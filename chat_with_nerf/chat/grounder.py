from typing import Callable
import math
import ast
import numpy as np
import open3d as o3d
import os
from copy import deepcopy
from chat_with_nerf import logger
from chat_with_nerf.chat.session import Session
from chat_with_nerf.settings import Settings

from chat_with_nerf.visual_grounder.captioner import BaseCaptioner
from chat_with_nerf.visual_grounder.picture_taker import PictureTaker
from chat_with_nerf.visual_grounder.visual_grounder import VisualGrounder


def ground(
    session: Session,
    dropdown_scene: str,
    ground_text: str,
    picture_taker: PictureTaker,
    captioner: None,
) -> list[tuple[str, str]] | None:
    """Ground a text in a scene by returning the relavant images and their
    corresponding captions.

    :param ground_text: the text query to be grounded
    :type ground_text: str
    :param visual_grounder: a visual grounder model
    :type visual_grounder: VisualGrounder
    :param captioner: a BaseCaptioner model
    :type captioner: BaseCaptioner
    """

    if Settings.USE_FAKE_GROUNDER:
        print("FAKE: ", Settings.USE_FAKE_GROUNDER)
        return [
            (
                "/workspace/chat-with-nerf/grounder_output/rgb/000.png",
                "a long sofa with white cover and yellow accent, metallic legs",
            ),
            (
                "/workspace/chat-with-nerf/grounder_output/rgb/001.png",
                "a loveseat with a pillow on top, white cover and yellow accent, metallic legs",
            ),
        ]

    logger.info(f"Ground Text: {ground_text}")
    # TODO: fix this!
    captioner_result, grounding_result_mesh_path = VisualGrounder.call_visual_grounder(
        session.session_id, ground_text, picture_taker, captioner
    )
    logger.debug(f"call_visual_grounder captioner_result: {captioner_result}")
    logger.debug(
        f"call_visual_grounder grounding_result_mesh_path: {grounding_result_mesh_path}"
    )

    session.grounding_result_mesh_path = grounding_result_mesh_path

    if captioner_result is None:
        return None

    result = []
    for img_path, img_caption in captioner_result.items():
        # Gradio uses http://localhost:7777/file=/absolute/path/example.jpg to access files,
        # can use relative too, just drop the leading slash
        result.append((img_path, img_caption))

    ##

    return result


def prettify_mesh_for_gradio(mesh):
    # Define the transformation matrix
    T = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

    # Apply the transformation
    mesh.transform(T)

    mesh.scale(10.0, center=mesh.get_center())

    bright_factor = 1  # Adjust this factor to get the desired brightness
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(mesh.vertex_colors) * bright_factor, 0, 1)
    )

    return mesh


def create_cylinder_mesh(p0, p1, color, radius=0.02, resolution=20, split=1):
    """Create a colored cylinder mesh between two points p0 and p1."""
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=1, resolution=resolution, split=split
    )
    transformation = cylinder_frame(p0, p1)
    cylinder.transform(transformation)
    # Apply color
    cylinder.paint_uniform_color(color)
    return cylinder


def cylinder_frame(p0, p1):
    """Calculate the transformation matrix to position a unit cylinder between two points."""
    direction = np.asarray(p1) - np.asarray(p0)
    length = np.linalg.norm(direction)
    direction /= length
    # Computing rotation matrix using Rodrigues' formula
    rot_axis = np.cross([0, 0, 1], direction)
    rot_angle = np.arccos(np.dot([0, 0, 1], direction))
    rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)

    # Translation
    translation = (np.asarray(p0) + np.asarray(p1)) / 2

    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix
    transformation[:3, 3] = translation
    scaling = np.eye(4)
    scaling[2, 2] = length
    transformation = np.matmul(transformation, scaling)
    return transformation


def create_bbox(center, extents, color=[1, 0, 0], radius=0.02):
    """Create a colored bounding box with given center, extents, and line thickness."""
    # ... [The same code as before to define corners and lines] ...
    sx, sy, sz = extents
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    cylinders = []
    for line in lines:
        p0, p1 = corners_3d[line[0]], corners_3d[line[1]]
        cylinders.append(create_cylinder_mesh(p0, p1, color, radius))
    return cylinders


def highlight_clusters_in_mesh(session, mesh) -> str:
    # Visualize the highlighted points by drawing 3D bounding boxes overlay on a mesh
    mesh = deepcopy(mesh)
    output_path = os.path.join(Settings.output_path, "mesh_vis")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if Settings.IS_SCANNET:
        mesh_file_path = os.path.join(output_path, f"{session.session_id}.obj")
    else:
        mesh_file_path = os.path.join(output_path, f"{session.session_id}.glb")

    # TODO: fix this!
    # mesh = o3d.io.read_triangle_mesh(session.base_mesh_path)
    candidates = session.candidate_visualization
    landmark = session.landmark_visualization
    top_5_objects2scores = session.top_5_objects2scores
    top_5_objects = list(top_5_objects2scores.keys())
    for candidate_id, candidate in enumerate(candidates):
        center = candidate["centroid"]
        extent = candidate["extent"]
        if candidate_id == session.chosen_candidate_id:
            bbox = create_bbox(center, extent, color=[0, 1, 0])
            for b in bbox:
                mesh += b
        else:
            if str(candidate_id) in top_5_objects:
                bbox = create_bbox(center, extent, color=[1, 0, 0])
                for b in bbox:
                    mesh += b

    if len(landmark) == 1:
        landmark_info = landmark[0]
        if not isinstance(landmark_info[0], list):
            centeroid = landmark_info[0].tolist()
        else:
            centeroid = landmark_info[0]
        bbox = create_bbox(centeroid, landmark_info[1], color=[0, 0, 1])
        for b in bbox:
            mesh += b

    mesh = prettify_mesh_for_gradio(mesh)
    o3d.io.write_triangle_mesh(mesh_file_path, mesh, write_vertex_colors=True)

    return mesh_file_path


def ground_with_gpt(
    session: Session,
    dropdown_scene: str,
    ground_json: str | dict,
    picture_taker: PictureTaker,
):
    print(f"{'*' * 100}\n{ground_json}\n{'*' * 100}")
    if isinstance(ground_json, str):
        ground_json = ast.literal_eval(ground_json)

    if session.working_scene_name.startswith("s"):
        landmark_location_list = {}
        target_data = ground_json.pop("target", None)
        if target_data:
            centroids, extends = VisualGrounder.target_finder_openscene(
                session, target_data["phrase"], picture_taker
            )
            round_target_bboxes = [
                {
                    "centroid": [round(ele, 1) for ele in c.tolist()],
                    "extent": [round(ele, 1) for ele in list(e)],
                }
                for c, e in zip(centroids, extends)
            ]
            full_target_bboxes = [
                {
                    "centroid": c.tolist(),
                    "extent": list(e),
                }
                for c, e in zip(centroids, extends)
            ]
        else:
            full_target_bboxes = []
            round_target_bboxes = []

        for value in ground_json.values():
            (
                landmark_location,
                landmark_extend,
            ) = VisualGrounder.landmark_finder_openscene(
                session, value["phrase"], picture_taker
            )
            landmark_location_list[value["phrase"]] = landmark_location

    else:
        landmark_location_list = {}

        target_data = ground_json.pop("target", None)
        if target_data:
            (centroids, extends), paths2images = VisualGrounder.target_finder(
                session, target_data["phrase"], picture_taker
            )
            round_target_bboxes = [
                {
                    "centroid": [round(ele, 1) for ele in c.tolist()],
                    "extent": [round(ele, 1) for ele in list(e)],
                }
                for c, e in zip(centroids, extends)
            ]
            full_target_bboxes = [
                {
                    "centroid": c.tolist(),
                    "extent": list(e),
                }
                for c, e in zip(centroids, extends)
            ]
        else:
            full_target_bboxes = []
            round_target_bboxes = []

        for key, value in ground_json.items():
            if key != "landmark" or value["phrase"] is None:
                continue
            landmark_location, landmark_extend = VisualGrounder.landmark_finder(
                session, value["phrase"], picture_taker
            )
            landmark_location_list[value["phrase"]] = landmark_location.tolist()

    # evaluation code is below which pass back to llm
    # TODO: compute the volume of the bounding box
    if len(landmark_location_list) > 0:
        landmark_location_centroid = next(iter(landmark_location_list.values()))

    if len(round_target_bboxes) == 0:
        raise ValueError("No target candidate found!")

    evaluation = {
        "Target Candidate BBox ('centroid': [cx, cy, cz], 'extent': [dx, dy, dz])": {
            str(i): bbox for i, bbox in enumerate(round_target_bboxes)
        },
        "Target Candidate BBox Volume (meter^3)": {
            str(i): round(bbox["extent"][0] * bbox["extent"][1] * bbox["extent"][2], 3)
            for i, bbox in enumerate(full_target_bboxes)
        },
    }
    if len(landmark_location_list) > 0:
        evaluation["Targe Candidate Distance to Landmark (meter)"] = {
            str(i): round(
                math.sqrt(
                    (bbox["centroid"][0] - landmark_location_centroid[0]) ** 2
                    + (bbox["centroid"][1] - landmark_location_centroid[1]) ** 2
                    + (bbox["centroid"][2] - landmark_location_centroid[2]) ** 2
                ),
                3,
            )
            for i, bbox in enumerate(full_target_bboxes)
        }
        evaluation["Landmark Location: (cx, cy, cz)"] = (
            {
                phrase: [round(ele, 1) for ele in landmark]
                for phrase, landmark in landmark_location_list.items()
            },
        )

    session.candidate = evaluation[
        "Target Candidate BBox ('centroid': [cx, cy, cz], 'extent': [dx, dy, dz])"
    ]

    session.candidate_visualization = full_target_bboxes
    if len(landmark_location_list) > 0:
        session.landmark_visualization = [
            (landmark_location_centroid, list(landmark_extend))
        ]
    else:
        session.landmark_visualization = []
    return str(evaluation)


def grond_no_gpt(
    session: Session,
    ground_text: str,
    picture_taker: PictureTaker,
):
    (
        center_list,
        box_size_list,
        values_list,
    ) = VisualGrounder.call_visual_grounder_no_gpt(session, ground_text, picture_taker)
    return center_list, box_size_list, values_list


def ground_no_gpt_with_callback(
    session: Session,
    ground_text: str,
    picture_taker: PictureTaker,
    callback: Callable[[list[tuple[str, str]] | None, Session], None],
):
    center_list, box_size_list, values_list = grond_no_gpt(
        session, ground_text, picture_taker
    )
    callback((center_list, box_size_list, values_list), session)


def ground_with_callback_with_gpt(
    session: Session,
    dropdown_scene: str,
    ground_text: str,
    picture_taker: PictureTaker,
    captioner: None,
    callback: Callable[[list[tuple[str, str]] | None, Session], None],
):
    result = ground_with_gpt(
        session,
        dropdown_scene,
        ground_text,
        picture_taker,
    )
    callback(result, session)


def ground_with_callback(
    session: Session,
    dropdown_scene: str,
    ground_text: str,
    picture_taker: PictureTaker,
    captioner: BaseCaptioner,
    callback: Callable[[list[tuple[str, str]] | None, Session], None],
):
    result = ground(
        session,
        dropdown_scene,
        ground_text,
        picture_taker,
        captioner,
    )
    callback(result, session)
