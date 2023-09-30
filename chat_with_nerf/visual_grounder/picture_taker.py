import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from uuid import uuid4
import clip
import h5py
import mediapy as media
import numpy as np
import open3d as o3d
import torch
import open_clip
from attrs import define
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks

# from nerfstudio.cameras.cameras import CameraType
from nerfstudio.utils.eval_utils import eval_setup
from sklearn.cluster import DBSCAN
from torch import Tensor

from transformers import AutoTokenizer, CLIPVisionModel

from chat_with_nerf import logger
from chat_with_nerf.chat.session import Session
from chat_with_nerf.model.scene_config import SceneConfig
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.camera_pose import CameraPose
from chat_with_nerf.visual_grounder.image_ref import ImageRef
from typing import Callable, Optional


@define
class PictureTaker:
    scene: str
    scene_config: SceneConfig
    lerf_pipeline: Pipeline
    h5_dict: dict
    clip_model: Optional[None]
    tokenizer: Optional[None]
    neg_embeds: Tensor
    negative_words_length: int
    thread_pool_executor: ThreadPoolExecutor
    openscene_embedding: Optional[np.ndarray]
    clip_preprocess: Optional[Callable]
    device: Optional[str]
    mesh: Optional[o3d.geometry.TriangleMesh]
    axis_align_matrix: Optional[np.ndarray]

    @staticmethod
    def render_picture(
        lerf_pipeline: Pipeline, camera_pose: dict, session_id: str
    ) -> ImageRef:
        logger.info("Picture Taking...")
        install_checks.check_ffmpeg_installed()
        camera = get_path_from_json(camera_pose)
        # camera_type = CameraType.PESPECTIVE
        camera.rescale_output_resolution(1.0)
        camera = camera.to(lerf_pipeline.device)
        output_filepath_path = Path(Settings.output_path) / session_id / "images"
        rgb_image_dir = output_filepath_path / "rgb"
        rgb_image_dir.mkdir(parents=True, exist_ok=True)

        result = {}
        camera_idx = 0
        aabb_box = None
        camera_ray_bundle = camera.generate_rays(
            camera_indices=camera_idx, aabb_box=aabb_box
        )
        with torch.no_grad():
            outputs = lerf_pipeline.model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle.to(lerf_pipeline.device)
            )

        output_image = outputs["rgb"].cpu().numpy()
        print(output_image.shape)

        if output_image.shape[-1] == 1:
            output_image = np.concatenate((output_image,) * 3, axis=-1)

        # saving rgb
        rgb = "rgb" + str(camera_idx)
        # create file name
        rgb_filename = rgb + "_" + str(uuid4()) + ".png"
        result[rgb] = str(rgb_image_dir) + "/" + rgb_filename
        media.write_image(result[rgb], output_image)

        imageRef = ImageRef(result[rgb], output_image)

        return imageRef

    def visual_ground_pipeline_no_gpt(self, query: str, session_id: str):
        prob_per_scale = self.compute_probability_query_property(query, session_id)
        best_scale_for_phrases: list[None | Tensor] = -1
        probability_per_scale_per_phrase: None | Tensor = None
        scales_list = torch.linspace(0.0, 1.5, 30)
        for i, scale in enumerate(scales_list):
            pos_prob = prob_per_scale[i]
            if (
                best_scale_for_phrases == -1
                or pos_prob.max() > probability_per_scale_per_phrase.max()  # type: ignore
            ):
                best_scale_for_phrases = scale
                probability_per_scale_per_phrase = pos_prob

        possibility_array = probability_per_scale_per_phrase.detach().cpu().numpy().squeeze()  # type: ignore # noqa: E501
        # if Settings.TOP_THREE_NO_GPT:'

        centroids_list, extends_list, values_list = self.find_clusters(
            possibility_array
        )
        # conrners_3d_list = []
        # for center, box_size in zip(center_list, box_size_list):
        #     conrners_3d_list.append(self.construct_bbox_corners(center, box_size))
        return centroids_list, extends_list, values_list

    def visual_ground_pipeline_with_gpt_lerf(self, query: str, session_id: str):
        prob_per_scale = self.compute_probability_query_property(query, session_id)
        best_scale_for_phrases: list[None | Tensor] = -1
        probability_per_scale_per_phrase: None | Tensor = None
        scales_list = torch.linspace(0.0, 1.5, 30)
        for i, scale in enumerate(scales_list):
            pos_prob = prob_per_scale[i]
            if (
                best_scale_for_phrases == -1
                or pos_prob.max() > probability_per_scale_per_phrase.max()  # type: ignore
            ):
                best_scale_for_phrases = scale
                probability_per_scale_per_phrase = pos_prob

        possibility_array = probability_per_scale_per_phrase.detach().cpu().numpy().squeeze()  # type: ignore # noqa: E501
        # if Settings.TOP_THREE_NO_GPT:'

        centroids_list, extends_list, values_list = self.find_clusters(
            possibility_array
        )
        # conrners_3d_list = []
        # for center, box_size in zip(center_list, box_size_list):
        #     conrners_3d_list.append(self.construct_bbox_corners(center, box_size))
        combined_list = list(zip(values_list, centroids_list, extends_list))
        sorted_list = sorted(
            combined_list, key=lambda x: x[0], reverse=True
        )  # reverse=True for descending order
        ordered_result = [(center, box_size) for _, center, box_size in sorted_list]
        best_center, box_size = ordered_result[0]
        return best_center, box_size

    def construct_bbox_corners(self, center, box_size):
        sx, sy, sz = box_size
        x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
        y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
        z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)

        return corners_3d

    def find_clusters(self, probability_over_all_points: np.ndarray):
        # Calculate the number of top values directly
        top_count = int(probability_over_all_points.size * 0.01)
        top_indices = np.argpartition(probability_over_all_points, -top_count)[
            -top_count:
        ]
        if Settings.IS_SCANNET:
            points_scannet = self.h5_dict["points_scannet"]
            # top_positions = points_scannet[top_indices]
            top_values = probability_over_all_points[top_indices].flatten()
            axis_align_matrix = self.axis_align_matrix
            pts = np.ones((points_scannet.shape[0], 4))
            pts[:, 0:3] = points_scannet[:, 0:3]
            pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
            aligned_vertices = np.copy(points_scannet)
            aligned_vertices[:, 0:3] = pts[:, 0:3]
            points_scannet = aligned_vertices

            top_positions_scannet = points_scannet[top_indices]
            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=0.05, min_samples=15)
            clusters = dbscan.fit(top_positions_scannet)
            labels = clusters.labels_

            # Initialize empty lists to store centroids and extends of each cluster
            centroids = []
            extends = []
            values_list = []

            for cluster_id in set(labels):
                if cluster_id == -1:  # Ignore noise
                    continue

                members = top_positions_scannet[labels == cluster_id]
                values = top_values[labels == cluster_id]
                centroid = np.mean(members, axis=0)
                mean_value = np.mean(values, axis=0)

                sx = np.max(members[:, 0]) - np.min(members[:, 0])
                sy = np.max(members[:, 1]) - np.min(members[:, 1])
                sz = np.max(members[:, 2]) - np.min(members[:, 2])

                # Append centroid and extends to the lists
                centroids.append(centroid)
                extends.append((sx, sy, sz))
                values_list.append(mean_value)
        else:
            if np.nonzero(probability_over_all_points > 0.50)[0].shape[0] == 0:
                logger.info("No points found for clustering.")
                return [], [], []
            logger.info(f"Selected {top_indices.shape[0]} points for clustering.")

            points = self.h5_dict["points"]
            origins = self.h5_dict["origins"]

            top_positions = points[top_indices]
            top_origins = origins[top_indices]
            top_values = probability_over_all_points[top_indices].flatten()

            logger.info("Clustering...")

            # Apply DBSCAN clustering
            epsilon = 0.05  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. # noqa: E501
            min_samples = int(15)  # Minimum number of samples in a cluster
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit(top_positions)

            labels = clusters.labels_

            logger.info(f"Found {len(set(labels))} clusters.")

            centroids = []
            extends = []
            values_list = []
            # Iterate over each cluster ID
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise
                    continue
                else:
                    # Compute the centroid of each cluster
                    members = top_positions[
                        labels == cluster_id
                    ]  # Get all members of the cluster
                    centroid = np.mean(members, axis=0)
                    centroids.append(centroid)
                    sx = np.max(members[:, 0]) - np.min(members[:, 0])
                    sy = np.max(members[:, 1]) - np.min(members[:, 1])
                    sz = np.max(members[:, 2]) - np.min(members[:, 2])
                    extends.append((sx, sy, sz))

                    valuess_for_members = top_values[labels == cluster_id]
                    mean_value = np.mean(valuess_for_members, axis=0)
                    values_list.append(mean_value)

        return centroids, extends, values_list

    def find_cluster(self, probability_over_all_points: np.ndarray):
        # Calculate the number of top values directly
        top_count = int(probability_over_all_points.size * 0.005)
        # print(top_count)
        # print(probability_over_all_points.size)
        # Find the indices of the top values
        top_indices = np.argpartition(probability_over_all_points, -top_count)[
            -top_count:
        ]
        # Fetch related data from the HDF5 dictionary
        points = self.h5_dict["points_scannet"]
        # origins = self.h5_dict["origins"]

        top_positions = points[top_indices]
        # top_origins = origins[top_indices]
        top_values = probability_over_all_points[top_indices].flatten()

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.05, min_samples=15)  # Directly use values where possible
        clusters = dbscan.fit(top_positions)
        labels = clusters.labels_

        # Find the cluster with the point closest to its centroid that has the highest value
        best_cluster_value = -np.inf
        best_cluster_id = None
        for cluster_id in set(labels):
            if cluster_id == -1:  # Ignore noise
                continue

            members = top_positions[labels == cluster_id]
            values_for_members = top_values[labels == cluster_id]

            # Calculate the centroid of the cluster
            centroid = np.mean(members, axis=0)

            # Compute the distance of all members to the centroid
            distances_to_centroid = np.linalg.norm(members - centroid, axis=1)

            # Find the index of the member closest to the centroid
            closest_member_idx = np.argmin(distances_to_centroid)

            # If this member has a better value than the current best, update
            if values_for_members[closest_member_idx] > best_cluster_value:
                best_cluster_value = values_for_members[closest_member_idx]
                best_cluster_id = cluster_id
        # For the best cluster, compute its centroid, bounding box and other desired values
        members_of_best_cluster = top_positions[labels == best_cluster_id]
        # values_for_best_cluster = top_values[labels == best_cluster_id]
        # origins_for_best_cluster = top_origins[labels == best_cluster_id]

        # Calculate the centroid of the best cluster
        centroid_of_best = np.mean(members_of_best_cluster, axis=0)

        # Determine the bounding box
        # min_bounds = np.min(members_of_best_cluster, axis=0)
        # max_bounds = np.max(members_of_best_cluster, axis=0)

        sx = np.max(members_of_best_cluster[:, 0]) - np.min(
            members_of_best_cluster[:, 0]
        )
        sy = np.max(members_of_best_cluster[:, 1]) - np.min(
            members_of_best_cluster[:, 1]
        )
        sz = np.max(members_of_best_cluster[:, 2]) - np.min(
            members_of_best_cluster[:, 2]
        )

        return centroid_of_best, (sx, sy, sz)

    def find_clusters_with_gpt(
        self,
        probability_over_all_points: np.ndarray,
        best_scale_for_phrases: float,
        session: Session,
    ):
        probability_over_all_points = (
            probability_over_all_points.detach().cpu().flatten().numpy()
        )
        top_count = int(probability_over_all_points.size * 0.005)
        top_indices = np.argpartition(probability_over_all_points, -top_count)[
            -top_count:
        ]

        # mesh_vertices = np.asarray(mesh.vertices)
        if session.working_scene_name.startswith("s"):
            points_scannet = self.h5_dict["points_scannet"]
            axis_align_matrix = self.axis_align_matrix
            pts = np.ones((points_scannet.shape[0], 4))
            pts[:, 0:3] = points_scannet[:, 0:3]
            pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
            aligned_vertices = np.copy(points_scannet)
            aligned_vertices[:, 0:3] = pts[:, 0:3]
            points_scannet = aligned_vertices
            # mesh.vertices = o3d.utility.Vector3dVector(aligned_vertices)
            points_nerfstudio = self.h5_dict["points_nerfstudio"]
            origins = self.h5_dict["origins"]
            top_positions_scannet = points_scannet[top_indices]
            top_values = probability_over_all_points[top_indices].flatten()
            top_position_nerfstudio = points_nerfstudio[top_indices]
            top_origins = origins[top_indices]

            dbscan = DBSCAN(eps=0.05, min_samples=15)
            clusters = dbscan.fit(top_positions_scannet)
            labels = clusters.labels_
            best_member_list = []
            origin_for_best_member_list = []
            centroids = []
            bboxes = []

            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise
                    continue

                # finding bbox in scannet coordination system
                cluster_members_scannet = top_positions_scannet[labels == cluster_id]
                centroid = np.mean(cluster_members_scannet, axis=0)
                centroids.append(centroid)

                # Calculate bounding box for the cluster
                sx = np.max(cluster_members_scannet[:, 0]) - np.min(
                    cluster_members_scannet[:, 0]
                )
                sy = np.max(cluster_members_scannet[:, 1]) - np.min(
                    cluster_members_scannet[:, 1]
                )
                sz = np.max(cluster_members_scannet[:, 2]) - np.min(
                    cluster_members_scannet[:, 2]
                )
                bboxes.append((sx, sy, sz))

                # finding best candidate in nerfstudio coordination system
                # if Settings.NO_VISUAL_FEEDBACK is False:
                valuess_for_members = top_values[labels == cluster_id]
                best_index = np.argmax(valuess_for_members)

                cluster_members_nerfstudio = top_position_nerfstudio[
                    labels == cluster_id
                ]
                origin_for_members = top_origins[labels == cluster_id]

                closest_member_to_centroid_nerfstudio = cluster_members_nerfstudio[
                    best_index
                ]
                best_member_list.append(closest_member_to_centroid_nerfstudio)
                origin_for_best_member_list.append(origin_for_members[best_index])
        else:
            if np.nonzero(probability_over_all_points > 0.50)[0].shape[0] == 0:
                logger.info("No points found for clustering.")
                return [], None
            logger.info(f"Selected {top_indices.shape[0]} points for clustering.")

            points = self.h5_dict["points"]
            origins = self.h5_dict["origins"]

            top_positions = points[top_indices]
            top_origins = origins[top_indices]
            top_values = probability_over_all_points[top_indices].flatten()

            logger.info("Clustering...")

            # Apply DBSCAN clustering
            epsilon = 0.05  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. # noqa: E501
            min_samples = int(15)  # Minimum number of samples in a cluster
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit(top_positions)

            labels = clusters.labels_

            logger.info(f"Found {len(set(labels))} clusters.")

            best_member_list = []
            origin_for_best_member_list = []
            centroids = []
            bboxes = []
            # Iterate over each cluster ID
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise
                    continue
                else:
                    # Compute the centroid of each cluster
                    members = top_positions[
                        labels == cluster_id
                    ]  # Get all members of the cluster
                    centroid = np.mean(members, axis=0)
                    centroids.append(centroid)
                    sx = np.max(members[:, 0]) - np.min(members[:, 0])
                    sy = np.max(members[:, 1]) - np.min(members[:, 1])
                    sz = np.max(members[:, 2]) - np.min(members[:, 2])
                    bboxes.append((sx, sy, sz))

                    valuess_for_members = top_values[labels == cluster_id]
                    orgins_for_this_cluster = top_origins[labels == cluster_id]

                    best_index = np.argmax(valuess_for_members, axis=0)

                    closest_member_to_centroid = members[best_index]
                    best_member_list.append(closest_member_to_centroid)

                    origin_of_best_member = orgins_for_this_cluster[best_index]
                    origin_for_best_member_list.append(origin_of_best_member)

        paths2images = []
        # if Settings.NO_VISUAL_FEEDBACK is False:
        if session.working_scene_name.startswith("s"):
            c2w_list = [
                self.compute_camera_to_world_matrix(
                    member, origin, best_scale_for_phrases
                )
                for member, origin in zip(best_member_list, origin_for_best_member_list)
            ]

            camera_pose_instance = CameraPose()
            camera_poses = [
                camera_pose_instance.construct_camera_pose(c2w) for c2w in c2w_list
            ]

            session.camera_poses = camera_poses
        else:
            c2w_list = [
                self.compute_camera_to_world_matrix(
                    member, origin, best_scale_for_phrases
                )
                for member, origin in zip(
                    best_member_list,
                    origin_for_best_member_list,
                )
            ]
            camera_pose_instance = CameraPose()
            camera_poses = [
                camera_pose_instance.construct_camera_pose(c2w) for c2w in c2w_list
            ]
            session.camera_poses = camera_poses
        return (centroids, bboxes), paths2images

    def take_picture_for_the_ground_result(self, session: Session, choosen_id: int):
        camera_poses = session.camera_poses
        # camera_pose = [camera_poses[choosen_id]]
        lerf_pipelines = [self.lerf_pipeline] * len(camera_poses)
        session_id_list = [session.session_id] * len(camera_poses)
        paths2images: Iterator[ImageRef] = self.thread_pool_executor.map(
            lambda tup: PictureTaker.render_picture(*tup),
            (
                (lerf_pipeline, camera_pose, session_id)
                for lerf_pipeline, camera_pose, session_id in zip(
                    lerf_pipelines, camera_poses, session_id_list
                )
            ),
        )
        return list(paths2images)

    def visual_ground_pipeline_with_gpt(self, positive_phrase: str, session: Session):
        prob_per_scale = self.compute_probability_query_property(
            positive_phrase, session.session_id
        )
        best_scale_for_phrases: None | float = -1
        probability_per_scale_per_phrase: None | Tensor = None
        scales_list = torch.linspace(0.0, 1.5, 30)
        for i, scale in enumerate(scales_list):
            pos_prob = prob_per_scale[i]
            if (
                best_scale_for_phrases == -1
                or pos_prob.max() > probability_per_scale_per_phrase.max()  # type: ignore
            ):
                best_scale_for_phrases = scale.item()
                probability_per_scale_per_phrase = pos_prob

        (centroids, bboxes), paths2images = self.find_clusters_with_gpt(
            probability_per_scale_per_phrase, best_scale_for_phrases, session
        )

        return (centroids, bboxes), paths2images

    def compute_probability_query_property(self, query: str, session: Session):
        positives = [query]
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in positives]
            ).to("cuda")
            pos_embeds = self.clip_model.encode_text(tok_phrases)
        pos_embeds /= pos_embeds.norm(dim=-1, keepdim=True)
        scales_list = torch.linspace(0.0, 1.5, 30)

        n_phrases = len(positives)
        prob_per_scale = []
        for index, _ in enumerate(scales_list):
            clip_output = torch.from_numpy(
                self.h5_dict["clip_embeddings_per_scale"][index]
            ).to("cuda")
            # TODO: ensure i = 1
            for i in range(n_phrases):
                probs = self.get_relevancy(
                    embed=clip_output,
                    positive_id=i,
                    pos_embeds=pos_embeds,
                    neg_embeds=self.neg_embeds,
                    positive_words_length=1,
                )
                pos_prob = probs[..., 0:1]
                # assert torch.sum(pos_prob) == 1
                prob_per_scale.append(pos_prob)

        return prob_per_scale

    def take_picture(
        self, query: str, session: Session
    ) -> tuple[list[ImageRef], str | None]:
        """Returns a list of ImageRef and the path to the grounding result mesh
        file."""
        positives = [query]
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in positives]
            ).to("cuda")
            pos_embeds = self.clip_model.encode_text(tok_phrases)
        pos_embeds /= pos_embeds.norm(dim=-1, keepdim=True)
        # use query to dot product with the point cloud -> centroids
        scales_list = torch.linspace(0.0, 1.5, 30)

        n_phrases = len(positives)
        best_scale_for_phrases: list[None | Tensor] = [None for _ in range(n_phrases)]
        probability_per_scale_per_phrase: list[None | Tensor] = [
            None for _ in range(n_phrases)
        ]
        for i, scale in enumerate(scales_list):
            clip_output = torch.from_numpy(
                self.h5_dict["clip_embeddings_per_scale"][i]
            ).to("cuda")
            for i in range(n_phrases):
                probs = self.get_relevancy(
                    embed=clip_output,
                    positive_id=i,
                    pos_embeds=pos_embeds,
                    neg_embeds=self.neg_embeds,
                    positive_words_length=1,
                )
                pos_prob = probs[..., 0:1]
                if (
                    best_scale_for_phrases[i] is None
                    or pos_prob.max() > probability_per_scale_per_phrase[i].max()  # type: ignore
                ):
                    best_scale_for_phrases[i] = scale
                    probability_per_scale_per_phrase[i] = pos_prob

        possibility_array = probability_per_scale_per_phrase[0].detach().cpu().numpy()  # type: ignore # noqa: E501
        # best_scale = best_scale_for_phrases[0].item()
        num_points = possibility_array.shape[0]
        percentage_points = int(num_points * 0.005)
        flattened_values = possibility_array.flatten()

        # # Find the indices of the top 0.5% values
        top_indices = np.argpartition(flattened_values, -percentage_points)[
            -percentage_points:
        ]

        # top_indices = np.nonzero(possibility_array > 0.55)[0]
        if np.nonzero(possibility_array > 0.55)[0].shape[0] == 0:
            logger.info("No points found for clustering.")
            return [], None

        logger.info(f"Selected {top_indices.shape[0]} points for clustering.")

        points = self.h5_dict["points"]
        origins = self.h5_dict["origins"]

        top_positions = points[top_indices]
        top_origins = origins[top_indices]
        top_values = possibility_array[top_indices].flatten()

        logger.info("Clustering...")

        # Apply DBSCAN clustering
        epsilon = 0.05  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. # noqa: E501
        min_samples = int(15)  # Minimum number of samples in a cluster
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        clusters = dbscan.fit(top_positions)

        labels = clusters.labels_

        logger.info(f"Found {len(set(labels))} clusters.")

        # Find the closest member to the centroid of each cluster
        # and use its origin to render the picture
        best_member_list = []
        origin_for_best_member_list = []
        # Iterate over each cluster ID
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise
                continue
            else:
                # Compute the centroid of each cluster
                members = top_positions[
                    labels == cluster_id
                ]  # Get all members of the cluster
                valuess_for_members = top_values[labels == cluster_id]
                orgins_for_this_cluster = top_origins[labels == cluster_id]

                best_index = np.argmax(valuess_for_members, axis=0)

                closest_member_to_centroid = members[best_index]
                best_member_list.append(closest_member_to_centroid)

                origin_of_best_member = orgins_for_this_cluster[best_index]
                origin_for_best_member_list.append(origin_of_best_member)

        assert best_scale_for_phrases[0] is not None
        c2w_list = [
            self.compute_camera_to_world_matrix(
                member, origin, best_scale_for_phrases[0].item()
            )
            for member, origin in zip(
                best_member_list,
                origin_for_best_member_list,
            )
        ]
        camera_pose_instance = CameraPose()
        camera_poses = [
            camera_pose_instance.construct_camera_pose(c2w) for c2w in c2w_list
        ]

        lerf_pipelines = [self.lerf_pipeline] * len(camera_poses)
        session_id_list = [session.session_id] * len(camera_poses)
        # camera pose -> render pictures
        picture_paths: Iterator[ImageRef] = self.thread_pool_executor.map(
            lambda tup: PictureTaker.render_picture(*tup),
            (
                (lerf_pipeline, camera_pose, session.session_id)
                for lerf_pipeline, camera_pose, session.session_id in zip(
                    lerf_pipelines, camera_poses, session_id_list
                )
            ),
        )

        # Visualize the highlighted points by drawing 3D bounding boxes overlay on a mesh
        logger.info(
            "Export RGB GLB files drawing 3D bounding boxes overlay on a mesh..."
        )
        mesh_file_path = self.highlight_clusters_in_mesh(
            session_id=session.session_id, labels=labels, top_positions=top_positions
        )

        return list(picture_paths), mesh_file_path

    def highlight_clusters_in_mesh(
        self, session_id: str, labels: np.ndarray, top_positions: np.ndarray
    ) -> str:
        # Visualize the highlighted points by drawing 3D bounding boxes overlay on a mesh
        output_path = os.path.join(Settings.output_path, "mesh_vis")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        mesh_file_path = os.path.join(output_path, f"{session_id}.glb")

        mesh = o3d.io.read_triangle_mesh(self.scene_config.nerf_exported_mesh_path)

        # Create mesh for the bounding box and set color to red
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise
                continue
            else:
                members = top_positions[
                    labels == cluster_id
                ]  # Get all members of the cluster
                centroid = members.mean(axis=0)  # Compute centroid
                furthest_distance = np.max(
                    np.linalg.norm(members - centroid, axis=1)
                )  # Compute furthest distance
                sphere = self.create_mesh_sphere(
                    centroid,
                    furthest_distance,
                    color=[0.0, 1.0, 0.0],  # color the sphere green
                )
                mesh += sphere

        mesh = self.prettify_mesh_for_gradio(mesh)
        o3d.io.write_triangle_mesh(mesh_file_path, mesh, write_vertex_colors=True)

        return mesh_file_path

    @staticmethod
    def prettify_mesh_for_gradio(mesh):
        # Define the transformation matrix
        T = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

        # Apply the transformation
        mesh.transform(T)

        mesh.scale(10.0, center=mesh.get_center())

        bright_factor = 1.5  # Adjust this factor to get the desired brightness
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(mesh.vertex_colors) * bright_factor, 0, 1)
        )

        return mesh

    @staticmethod
    def create_mesh_sphere(center, radius, color=[0.0, 1.0, 0.0], resolution=30):
        # Create a unit sphere (radius 1, centered at origin)
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=1.0, resolution=resolution
        )

        # Scale to the desired radius
        mesh_sphere.scale(radius, center=(0, 0, 0))

        # Translate to the desired center
        mesh_sphere.translate(center)

        # Paint it with the desired color
        mesh_sphere.paint_uniform_color(color)

        return mesh_sphere

    def get_relevancy(
        self,
        embed: torch.Tensor,
        positive_id: int,
        pos_embeds: Tensor,
        neg_embeds: Tensor,
        positive_words_length: int,
    ) -> torch.Tensor:
        phrases_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # noqa E501
        negative_vals = output[..., positive_words_length:]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(
            1, self.negative_words_length
        )  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(
                best_id.shape[0], self.negative_words_length, 2
            ),
        )[:, 0, :]

    def compute_camera_to_world_matrix(
        self, point: np.ndarray, origin: np.ndarray, k: float
    ) -> np.ndarray:
        epsilon = 1e-6
        direction = point - origin
        direction = direction / np.linalg.norm(direction)

        camera_position = point - (k * direction)

        up = np.array([0, 1, 0])

        right = np.cross(direction, up)
        right /= np.linalg.norm(right) + epsilon

        new_up = np.cross(right, direction)
        new_up /= np.linalg.norm(new_up) + epsilon

        camera_to_world = np.eye(4)
        camera_to_world[:3, 0] = right
        camera_to_world[:3, 1] = new_up
        camera_to_world[:3, 2] = -direction
        camera_to_world[:3, 3] = camera_position

        return camera_to_world.flatten()

    def find_clusters_openscene(self, vertices: np.ndarray, similarity: np.ndarray):
        # Calculate the number of top values directly
        top_positions = vertices
        # top_values = probability_over_all_points[top_indices].flatten()

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.05, min_samples=15)
        clusters = dbscan.fit(top_positions)
        labels = clusters.labels_

        # Initialize empty lists to store centroids and extends of each cluster
        centroids = []
        extends = []
        similarity_mean_list = []

        for cluster_id in set(labels):
            if cluster_id == -1:  # Ignore noise
                continue

            members = top_positions[labels == cluster_id]
            similarity_values = similarity[labels == cluster_id]
            simiarity_mean = np.mean(similarity_values)
            centroid = np.mean(members, axis=0)

            sx = np.max(members[:, 0]) - np.min(members[:, 0])
            sy = np.max(members[:, 1]) - np.min(members[:, 1])
            sz = np.max(members[:, 2]) - np.min(members[:, 2])

            # Append centroid and extends to the lists
            centroids.append(centroid)
            extends.append((sx, sy, sz))
            similarity_mean_list.append(simiarity_mean)

        return centroids, extends, similarity_mean_list

    def find_clusters_openscene_best(
        self, vertices: np.ndarray, similarity: np.ndarray
    ):
        # Calculate the number of top values directly
        top_positions = vertices
        # top_values = probability_over_all_points[top_indices].flatten()

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.05, min_samples=15)
        clusters = dbscan.fit(top_positions)
        labels = clusters.labels_

        # Initialize empty lists to store centroids and extends of each cluster
        centroids = []
        extends = []
        similarity_mean_list = []

        for cluster_id in set(labels):
            if cluster_id == -1:  # Ignore noise
                continue

            members = top_positions[labels == cluster_id]
            similarity_values = similarity[labels == cluster_id]
            simiarity_mean = np.mean(similarity_values)
            centroid = np.mean(members, axis=0)

            sx = np.max(members[:, 0]) - np.min(members[:, 0])
            sy = np.max(members[:, 1]) - np.min(members[:, 1])
            sz = np.max(members[:, 2]) - np.min(members[:, 2])

            # Append centroid and extends to the lists
            centroids.append(centroid)
            extends.append((sx, sy, sz))
            similarity_mean_list.append(simiarity_mean)

        seletec_idx = np.argmax(np.array(similarity_mean_list))
        return centroids[seletec_idx], extends[seletec_idx]

    def visual_ground_target_finder_with_gpt_openscene(
        self, positive_phrase: str, session_id: str
    ):
        text = clip.tokenize([positive_phrase]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)

        clip_embedding_numpy = torch.from_numpy(self.openscene_embedding)
        clip_embedding_numpy = clip_embedding_numpy.to(self.device)
        text_features = text_features.float()
        clip_embedding_numpy /= clip_embedding_numpy.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = clip_embedding_numpy @ text_features.T
        similarity = similarity.cpu().numpy()
        similarity = similarity.squeeze()
        turning_point = np.percentile(similarity, 90)
        mask = similarity > turning_point
        similarity = similarity[mask]
        # cmap = cm.get_cmap('viridis')
        # norm = plt.Normalize(-1, 1)
        # colored_values = cmap(norm(similarity))
        # colored_values = colored_values.squeeze()
        vertices = np.asarray(self.mesh.vertices)
        vertices = vertices[mask]
        centroids, extents, similarity_mean_list = self.find_clusters_openscene(
            vertices, similarity
        )
        return centroids, extents, similarity_mean_list

    def visual_ground_landmark_finder_with_gpt_openscene(
        self, positive_phrase: str, session_id: str
    ):
        text = clip.tokenize([positive_phrase]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)

        clip_embedding_numpy = torch.from_numpy(self.openscene_embedding)
        clip_embedding_numpy = clip_embedding_numpy.to(self.device)
        text_features = text_features.float()
        clip_embedding_numpy /= clip_embedding_numpy.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = clip_embedding_numpy @ text_features.T
        similarity = similarity.cpu().numpy()
        similarity = similarity.squeeze()
        turning_point = np.percentile(similarity, 95)
        mask = similarity > turning_point
        similarity = similarity[mask]
        # cmap = cm.get_cmap('viridis')
        # norm = plt.Normalize(-1, 1)
        # colored_values = cmap(norm(similarity))
        # colored_values = colored_values.squeeze()
        vertices = np.asarray(self.mesh.vertices)
        vertices = vertices[mask]
        centroid, extent = self.find_clusters_openscene_best(vertices, similarity)
        return centroid, extent


class PictureTakerFactory:
    picture_taker_dict: Optional[dict[str, PictureTaker]] = None

    @classmethod
    def get_picture_takers(
        cls, scene_configs: dict[str, SceneConfig]
    ) -> dict[str, PictureTaker]:
        return PictureTakerFactory.initialize_picture_takers(scene_configs)

    @classmethod
    def get_picture_takers_no_visual_feedback(
        cls, scene_configs: dict[str, SceneConfig]
    ) -> dict[str, PictureTaker]:
        return PictureTakerFactory.initialize_picture_takers_no_visual_feedback(
            scene_configs
        )

    @classmethod
    def get_picture_takers_no_gpt(
        cls, scene_configs: dict[str, SceneConfig]
    ) -> dict[str, PictureTaker]:
        return PictureTakerFactory.initialize_picture_takers_no_gpt(scene_configs)

    @classmethod
    def get_picture_takers_no_visual_feedback_openscene(
        cls, scene_configs: dict[str, SceneConfig]
    ) -> dict[str, PictureTaker]:
        if cls.picture_taker_dict is None:
            selected_configs_scannet = {
                k: v for k, v in scene_configs.items() if k.startswith("s")
            }
            selected_configs_inthewild = {
                k: v for k, v in scene_configs.items() if not k.startswith("s")
            }
            picture_taker_dict_inthewild = {}
            picture_taker_dict_scannet = {}
            if selected_configs_scannet:
                picture_taker_dict_inthewild = PictureTakerFactory.initialize_picture_takers_no_visual_feedback_openscene(
                    selected_configs_scannet
                )
            if selected_configs_inthewild:
                picture_taker_dict_scannet = (
                    PictureTakerFactory.initialize_picture_takers_no_gpt(
                        selected_configs_inthewild
                    )
                )
            combined_dict = {
                **picture_taker_dict_inthewild,
                **picture_taker_dict_scannet,
            }
            cls.picture_taker_dict = combined_dict
        return cls.picture_taker_dict

    @staticmethod
    def initialize_picture_takers_no_visual_feedback_openscene(
        scene_configs: dict[str, SceneConfig],
    ) -> dict[str, PictureTaker]:
        """_summary_

        Args:
            scene_configs (dict[str, SceneConfig]): _description_

        Returns:
            dict[str, PictureTaker]: _description_
        """
        picture_taker_dict = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14@336px", device=device)

        for scene_name, scene_config in scene_configs.items():
            openscene_embedding = PictureTakerFactory.load_openscene(
                scene_config.load_openscene
            )
            scene_mesh, axis_align_matrix = PictureTakerFactory.load_mesh(
                scene_config.load_mesh, scene_config.load_metadata
            )
            thread_pool_executor = ThreadPoolExecutor(max_workers=Settings.MAX_WORKERS)

            picture_taker_dict[scene_name] = PictureTaker(
                scene=scene_config.scene_name,
                scene_config=scene_config,
                lerf_pipeline=None,
                h5_dict=None,
                clip_model=model,
                tokenizer=None,
                neg_embeds=None,
                negative_words_length=0,
                thread_pool_executor=thread_pool_executor,
                openscene_embedding=openscene_embedding,
                clip_preprocess=preprocess,
                mesh=scene_mesh,
                device=device,
                axis_align_matrix=axis_align_matrix,
            )

        return picture_taker_dict

    @staticmethod
    def load_openscene(load_openscene: str) -> np.ndarray:
        openscene_emebdding = np.load(load_openscene)
        return openscene_emebdding

    @staticmethod
    def load_mesh(load_mesh: str, load_meta_file: str):
        mesh = o3d.io.read_triangle_mesh(load_mesh)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        axis_align_matrix = None
        axisAlignment_matrix = PictureTakerFactory.get_transformation_matrix(
            load_meta_file
        )
        mesh_vertices = np.asarray(mesh.vertices)
        axis_align_matrix = np.array(axisAlignment_matrix).reshape((4, 4))
        pts = np.ones((mesh_vertices.shape[0], 4))
        pts[:, 0:3] = mesh_vertices[:, 0:3]
        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
        aligned_vertices = np.copy(mesh_vertices)
        aligned_vertices[:, 0:3] = pts[:, 0:3]
        mesh.vertices = o3d.utility.Vector3dVector(aligned_vertices)
        return mesh, axis_align_matrix

    @staticmethod
    def load_inthewild_mesh(load_mesh: str):
        mesh = o3d.io.read_triangle_mesh(load_mesh)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        return mesh

    @staticmethod
    def get_transformation_matrix(meta_file):
        lines = open(meta_file).readlines()
        for line in lines:
            if "axisAlignment" in line:
                axis_align_matrix = [
                    float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
                ]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        return axis_align_matrix

    @staticmethod
    def initialize_picture_takers_no_visual_feedback(
        scene_configs: dict[str, SceneConfig],
    ) -> dict[str, PictureTaker]:
        picture_taker_dict = {}
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",  # e.g., ViT-B-16
            pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        model = model.to("cuda")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        negatives = ["object", "things", "stuff", "texture"]
        with torch.no_grad():
            tok_phrases = torch.cat([tokenizer(phrase) for phrase in negatives]).to(
                "cuda"
            )
            neg_embeds = model.encode_text(tok_phrases)
        neg_embeds /= neg_embeds.norm(dim=-1, keepdim=True)
        for scene_name, scene_config in scene_configs.items():
            h5_dict = PictureTakerFactory.load_h5_file(scene_config.load_h5_config)
            thread_pool_executor = ThreadPoolExecutor(max_workers=Settings.MAX_WORKERS)
            scene_mesh, axis_align_matrix = PictureTakerFactory.load_mesh(
                scene_config.load_mesh, scene_config.load_metadata
            )
            lerf_pipeline = PictureTakerFactory.initialize_lerf_pipeline(
                scene_config.load_lerf_config, scene_name
            )
            picture_taker_dict[scene_name] = PictureTaker(
                scene=scene_config.scene_name,
                scene_config=scene_config,
                lerf_pipeline=lerf_pipeline,
                h5_dict=h5_dict,
                clip_model=model,
                tokenizer=tokenizer,
                neg_embeds=neg_embeds,
                negative_words_length=len(negatives),
                thread_pool_executor=thread_pool_executor,
                openscene_embedding=None,
                clip_preprocess=None,
                mesh=scene_mesh,
                device=None,
                axis_align_matrix=axis_align_matrix,
            )

        return picture_taker_dict

    @staticmethod
    def initialize_picture_takers_no_gpt(
        scene_configs: dict[str, SceneConfig],
    ) -> dict[str, PictureTaker]:
        picture_taker_dict = {}
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",  # e.g., ViT-B-16
            pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        model = model.to("cuda")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        negatives = ["object", "things", "stuff", "texture"]
        with torch.no_grad():
            tok_phrases = torch.cat([tokenizer(phrase) for phrase in negatives]).to(
                "cuda"
            )
            neg_embeds = model.encode_text(tok_phrases)
        neg_embeds /= neg_embeds.norm(dim=-1, keepdim=True)
        for scene_name, scene_config in scene_configs.items():
            h5_dict = PictureTakerFactory.load_h5_file(scene_config.load_h5_config)
            mesh = PictureTakerFactory.load_inthewild_mesh(scene_config.load_mesh)
            thread_pool_executor = ThreadPoolExecutor(max_workers=Settings.MAX_WORKERS)
            lerf_pipeline = PictureTakerFactory.initialize_lerf_pipeline(
                scene_config.load_lerf_config, scene_name
            )
            picture_taker_dict[scene_name] = PictureTaker(
                scene=scene_config.scene_name,
                scene_config=scene_config,
                lerf_pipeline=lerf_pipeline,
                h5_dict=h5_dict,
                clip_model=model,
                tokenizer=tokenizer,
                neg_embeds=neg_embeds,
                negative_words_length=len(negatives),
                thread_pool_executor=thread_pool_executor,
                openscene_embedding=None,
                clip_preprocess=None,
                device=None,
                mesh=mesh,
                axis_align_matrix=None,
            )

        return picture_taker_dict

    @staticmethod
    def initialize_picture_takers(
        scene_configs: dict[str, SceneConfig],
    ) -> dict[str, PictureTaker]:
        picture_taker_dict = {}
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",  # e.g., ViT-B-16
            pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        model = model.to("cuda")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        negatives = ["object", "things", "stuff", "texture"]
        with torch.no_grad():
            tok_phrases = torch.cat([tokenizer(phrase) for phrase in negatives]).to(
                "cuda"
            )
            neg_embeds = model.encode_text(tok_phrases)
        neg_embeds /= neg_embeds.norm(dim=-1, keepdim=True)

        for scene_name, scene_config in scene_configs.items():
            lerf_pipeline = PictureTakerFactory.initialize_lerf_pipeline(
                scene_config.load_lerf_config, scene_name
            )
            h5_dict = PictureTakerFactory.load_h5_file(scene_config.load_h5_config)
            thread_pool_executor = ThreadPoolExecutor(max_workers=Settings.MAX_WORKERS)
            picture_taker_dict[scene_name] = PictureTaker(
                scene=scene_config.scene_name,
                scene_config=scene_config,
                lerf_pipeline=lerf_pipeline,
                h5_dict=h5_dict,
                clip_model=model,
                tokenizer=tokenizer,
                neg_embeds=neg_embeds,
                negative_words_length=len(negatives),
                thread_pool_executor=thread_pool_executor,
            )

        return picture_taker_dict

    @staticmethod
    def initialize_lerf_pipeline(load_config: str, scene_name: str) -> Pipeline:
        initial_dir = os.getcwd()
        print(str(Settings.NERF_DATA_PATH + "/" + scene_name))
        os.chdir(Settings.NERF_DATA_PATH + "/" + scene_name)
        _, lerf_pipeline, _, _ = eval_setup(
            Path(load_config),
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        os.chdir(initial_dir)
        return lerf_pipeline

    @staticmethod
    def load_h5_file(load_config: str) -> dict:
        print(load_config)
        hdf5_file = h5py.File(load_config, "r")
        # batch_idx = 5
        points = hdf5_file["points"]["points"][:]
        origins = hdf5_file["origins"]["origins"][:]
        directions = hdf5_file["directions"]["directions"][:]

        clip_embeddings_per_scale = []

        clips_group = hdf5_file["clip"]
        for i in range(30):
            clip_embeddings_per_scale.append(clips_group[f"scale_{i}"][:])

        rgb = hdf5_file["rgb"]["rgb"][:]
        hdf5_file.close()
        h5_dict = {
            "points": points,
            "origins": origins,
            "directions": directions,
            "clip_embeddings_per_scale": clip_embeddings_per_scale,
            "rgb": rgb,
        }
        return h5_dict
