import numpy as np


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
