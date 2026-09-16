"""Rotation helpers shared by adapters (numpy / scipy only)."""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def _gram_schmidt(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + EPS)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # columns b1,b2,b3


# -- 6-D rotation, COLUMN-STACKED layout [R[:,0], R[:,1]]  (X-VLA LIBERO client) -- #
def mat_to_rot6d_cols(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[..., :3, 0], R[..., :3, 1]], axis=-1)


def rot6d_cols_to_mat(r6: np.ndarray) -> np.ndarray:
    r6 = np.asarray(r6, dtype=np.float64)
    return _gram_schmidt(r6[..., 0:3], r6[..., 3:6])


# -- 6-D rotation, ROW-MAJOR INTERLEAVED layout R[:, :2].reshape(6) (X-VLA SIMPLER/CALVIN/VLABench) -- #
def mat_to_rot6d_rows(R: np.ndarray) -> np.ndarray:
    return np.asarray(R)[..., :, :2].reshape(*np.asarray(R).shape[:-2], 6)


def rot6d_rows_to_mat(r6: np.ndarray) -> np.ndarray:
    r6 = np.asarray(r6, dtype=np.float64)
    return _gram_schmidt(r6[..., 0:5:2], r6[..., 1:6:2])


# -- conversions via scipy -- #
def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


def mat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_euler("xyz")


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


def euler_xyz_to_mat(e: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", e, degrees=False).as_matrix()


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(q).as_matrix()


def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(v).as_matrix()


def robosuite_mat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """Exactly what the X-VLA LIBERO client does (robosuite mat2quat -> quat2axisangle)
    when robosuite is importable; scipy rotvec otherwise (same rotation)."""
    try:
        import robosuite.utils.transform_utils as T  # type: ignore

        return np.asarray(T.quat2axisangle(T.mat2quat(np.asarray(R, dtype=np.float64))))
    except Exception:
        return mat_to_rotvec(R)
