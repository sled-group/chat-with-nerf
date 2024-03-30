from sklearn.cluster import DBSCAN
import numpy as np
import clip
import torch
import json
import os


def find_clusters(vertices: np.ndarray, similarity: np.ndarray):
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


def ground_open_scene_embedding(query: str, device, model, clip_embedding, mesh):
    text = clip.tokenize([query], context_length=77, truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    clip_embedding_numpy = torch.from_numpy(clip_embedding)
    clip_embedding_numpy = clip_embedding_numpy.to(device)
    text_features = text_features.float()
    clip_embedding_numpy /= clip_embedding_numpy.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = clip_embedding_numpy @ text_features.T
    similarity = similarity.cpu().numpy()
    similarity = similarity.squeeze()
    turning_point = np.percentile(similarity, 95)
    mask = similarity > turning_point
    similarity = similarity[mask]
    vertices = np.asarray(mesh.vertices)
    vertices = vertices[mask]
    centroids, extents, similarity_mean_list = find_clusters(vertices, similarity)
    return centroids, extents, similarity_mean_list


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


def is_label_unique(data_list, target_label):
    encountered_labels = set()

    for entry in data_list:
        label = entry["label"]
        if label in encountered_labels:
            if label == target_label:
                return False
        else:
            encountered_labels.add(label)

    return True


def construct_bbox_corners(center, box_size):
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


def get_box3d_min_max(corner):
    """Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    """

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    """Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    """
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = (
        np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    )
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def process_json(data):
    # Your processing logic here
    return data["grounding_query"]


def process_all_json_files(directory):
    """
    Loads and processes all JSON files in the specified directory.

    :param directory: Path to the directory containing JSON files
    """
    # Ensure the provided directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    descriprion = []
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file is a JSON file
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            # Load the JSON file
            try:
                with open(filepath, "r") as file:
                    data = json.load(file)

                # Process the loaded JSON data
                sentence = process_json(data)
                descriprion.append(sentence)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return descriprion


def convert_origin_bbox(bbox, axisAlignment_matrix):
    center_aligned = np.append(np.array(bbox[:3]), 1)
    extents_aligned = np.array(bbox[3:])

    half_extents = extents_aligned / 2
    corners_aligned = [
        center_aligned[:3] + [-half_extents[0], -half_extents[1], -half_extents[2]],
        center_aligned[:3] + [half_extents[0], -half_extents[1], -half_extents[2]],
        center_aligned[:3] + [-half_extents[0], half_extents[1], -half_extents[2]],
        center_aligned[:3] + [half_extents[0], half_extents[1], -half_extents[2]],
        center_aligned[:3] + [-half_extents[0], -half_extents[1], half_extents[2]],
        center_aligned[:3] + [half_extents[0], -half_extents[1], half_extents[2]],
        center_aligned[:3] + [-half_extents[0], half_extents[1], half_extents[2]],
        center_aligned[:3] + [half_extents[0], half_extents[1], half_extents[2]],
    ]

    inverse_matrix = np.linalg.inv(axisAlignment_matrix)
    corners_original = [
        np.dot(inverse_matrix, np.append(corner, 1.0))[:3].tolist()
        for corner in corners_aligned
    ]
    center_original = np.mean(corners_original, axis=0).tolist()

    extents_original = [
        np.max([corner[0] for corner in corners_original])
        - np.min([corner[0] for corner in corners_original]),
        np.max([corner[1] for corner in corners_original])
        - np.min([corner[1] for corner in corners_original]),
        np.max([corner[2] for corner in corners_original])
        - np.min([corner[2] for corner in corners_original]),
    ]

    return corners_original, center_original, extents_original
