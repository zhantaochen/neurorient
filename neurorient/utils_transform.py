from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import numpy
import torch

"""
taken from https://scikit-surgerycore.readthedocs.io/en/stable/_modules/sksurgerycore/algorithms/averagequaternions.html#average_quaternions
"""
def average_quaternions(quaternions):
    """
    Calculate average quaternion

    :params quaternions: is a Nx4 torch tensor and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    """

    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = torch.zeros(size=(4, 4)).double()

    for i in range(0, samples):
        quat = quaternions[i, :]
        # multiply quat with its transposed version quat' and add mat_a
        mat_a = torch.outer(quat, quat) + mat_a

    # scale
    mat_a = (1.0/ samples)*mat_a
    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = torch.linalg.eig(mat_a)
    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.real.argsort(descending=True)]
    # return the real part of the largest eigenvector (has only real part)
    return torch.real(torch.ravel(eigen_vectors[:, 0]))


def weighted_average_quaternions(quaternions, weights):
    """
    Average multiple quaternions with specific weights

    :params quaternions: is a Nx4 torch tensor and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :params weights: The weight vector w must be of the same length as
        the number of rows in the

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    :raises: ValueError if all weights are zero
    """
    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = torch.zeros(size=(4, 4)).double()
    weight_sum = 0

    for i in range(0, samples):
        quat = quaternions[i, :]
        mat_a = weights[i] * torch.outer(quat, quat) + mat_a
        weight_sum += weights[i]

    if weight_sum <= 0.0:
        raise ValueError("At least one weight must be greater than zero")

    # scale
    mat_a = (1.0/weight_sum) * mat_a

    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = torch.linalg.eig(mat_a)

    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.real.argsort(descending=True)]

    # return the real part of the largest eigenvector (has only real part)
    return torch.real(torch.ravel(eigen_vectors[:, 0]))

