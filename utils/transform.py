import numpy as np

from scipy.spatial.transform import Rotation as R


def inv_transform(T):
    T_inv = np.identity(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -np.dot(T[:3, :3].T, T[:3, 3])
    return T_inv


def rotation(T):
    rm = T[:3, :3].tolist()
    return R.from_matrix(rm).as_euler('zyx', degrees=True)


def translation(T):
    return np.array([T[0][3], T[1][3], T[2][3]])


def calc_error(T1, T2):
    e1 = np.mean(np.abs(rotation(T1) - rotation(T2)))
    e2 = np.mean(np.abs(translation(T1) - translation(T2)))
    return e1, e2


def check(T1, T2, max_r=5, max_t=1):
    er, et = calc_error(T1, T2)
    # print(f"Rotation error: {er:.3f}, Translation error: {et:.3f}")
    return er < max_r and et < max_t


def rotate_transformation_matrix(T, rx, ry, rz):
    # Convert degrees to radians
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    RY = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    RZ = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return np.dot(np.dot(np.dot(T, RZ), RY), RX)