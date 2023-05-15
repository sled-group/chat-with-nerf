import os

import numpy as np
from requests import Response


def list_dirs(root_dir):
    dirs_to_return = []
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                dirs_to_return.append(entry.name)
    return dirs_to_return


def rotate_x(angle_degrees: int, c2w: np.ndarray) -> np.ndarray:
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1],
        ]
    )
    return c2w @ rotation_matrix


def rotate_y(angle_degrees: int, c2w: np.ndarray) -> np.ndarray:
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1],
        ]
    )
    return c2w @ rotation_matrix


def rotate_z(angle_degrees: int, c2w: np.ndarray) -> np.ndarray:
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return c2w @ rotation_matrix


def get_status_code_and_reason(response: Response) -> str:
    return f"{response.status_code} - {response.reason}"
