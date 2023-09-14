import numpy as np
import skopi as sk
from .config import xp, ndimage

"""
Batched versions of skopi geometry functions for handling rotations.
"""

def quaternion2rot3d(quat):
    """
    Convert a set of quaternions to rotation matrices.
    This function was originally adopted from:
    https://github.com/duaneloh/Dragonfly/blob/master/src/interp.c
    and then from skopi and has been modified to convert in batch.

    Parameters
    ----------
    quat : numpy.ndarray, shape (n_quat, 4)
        series of quaternions

    Returns
    -------
    rotation : numpy.ndarray, shape (n_quat, 3, 3)
        series of rotation matrices
    """
    q01 = quat[:, 0] * quat[:, 1]
    q02 = quat[:, 0] * quat[:, 2]
    q03 = quat[:, 0] * quat[:, 3]
    q11 = quat[:, 1] * quat[:, 1]
    q12 = quat[:, 1] * quat[:, 2]
    q13 = quat[:, 1] * quat[:, 3]
    q22 = quat[:, 2] * quat[:, 2]
    q23 = quat[:, 2] * quat[:, 3]
    q33 = quat[:, 3] * quat[:, 3]

    # Obtain the rotation matrix
    rotation = xp.zeros((quat.shape[0], 3, 3))
    rotation[:, 0, 0] = (1. - 2. * (q22 + q33))
    rotation[:, 0, 1] = 2. * (q12 - q03)
    rotation[:, 0, 2] = 2. * (q13 + q02)
    rotation[:, 1, 0] = 2. * (q12 + q03)
    rotation[:, 1, 1] = (1. - 2. * (q11 + q33))
    rotation[:, 1, 2] = 2. * (q23 - q01)
    rotation[:, 2, 0] = 2. * (q13 - q02)
    rotation[:, 2, 1] = 2. * (q23 + q01)
    rotation[:, 2, 2] = (1. - 2. * (q11 + q22))

    return rotation

def axis_angle_to_quaternion(axis, theta):
    """
    Convert an angular rotation around an axis series to quaternions.

    Parameters
    ----------
    axis : numpy.ndarray, size (num_pts, 3)
        axis vector defining rotation
    theta : numpy.ndarray, size (num_pts)
        angle in radians defining anticlockwise rotation around axis

    Returns
    -------
    quat : numpy.ndarray, size (num_pts, 4)
        quaternions corresponding to axis/theta rotations
    """
    axis /= xp.linalg.norm(axis, axis=1)[:, None]
    angle = theta / 2

    quat = xp.zeros((len(theta), 4))
    quat[:, 0] = xp.cos(angle)
    quat[:, 1:] = xp.sin(angle)[:, None] * axis

    return quat

def quaternion_product(q1, q0):
    """
    Compute quaternion product, q1 x q0, according to:
    https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html.
    This should yield the same result as pytorch3d.transforms.quaternion_multiply
    (or possibly the -1*q_prod, which represents the same rotation).

    Parameters
    ----------
    q0 : numpy.ndarray, shape (n, 4)
        first quaternion to rotate by
    q1 : numpy.ndarray, shape (n, 4)
        second quaternion to rotate by

    Returns
    -------
    q_prod : numpy.ndarray, shape (n, 4)
        quaternion product q1 x q0, rotation by q0 followed by q1
    """
    p0, p1, p2, p3 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    r0, r1, r2, r3 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
    q_prod = xp.array([r0 * p0 - r1 * p1 - r2 * p2 - r3 * p3,
                       r0 * p1 + r1 * p0 - r2 * p3 + r3 * p2,
                       r0 * p2 + r1 * p3 + r2 * p0 - r3 * p1,
                       r0 * p3 - r1 * p2 + r2 * p1 + r3 * p0]).T

    return q_prod

def get_preferred_orientation_quat(num_pts, sigma, base_quat=None):
    """
    Sample quaternions distributed around a given or random position in a restricted
    range in SO(3), where the spread of the distribution is determined by sigma.

    Parameters
    ----------
    num_pts : int
        number of quaternions to generate
    sigma : float
        standard deviation in radians for angular sampling
    base_quat : numpy.ndarray, size (4)
        quaternion about which to distribute samples, random if None

    Returns
    -------
    quat : numpy.ndarray, size (num_quats, 4)
        quaternions with preferred orientations
    """
    if base_quat is None:
        base_quat = xp.array(sk.get_random_quat(1))  # need to change to skopi
    base_quat = xp.tile(base_quat, num_pts).reshape(num_pts, 4)

    # need to change to skopi
    R_random = quaternion2rot3d(xp.array(sk.get_random_quat(num_pts)))
    unitvec = xp.array([0, 0, 1.0])
    rot_axis = xp.matmul(unitvec, R_random)
    theta = sigma * xp.random.randn(num_pts)
    rand_axis = theta[:, None] * rot_axis
    pref_quat = axis_angle_to_quaternion(rand_axis, theta)
    quat = quaternion_product(pref_quat, base_quat)

    return quat
