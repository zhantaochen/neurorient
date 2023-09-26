import unittest
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import numpy
import torch

from .reconstruction.slicing import gen_nonuniform_normalized_positions

import cupy as xp

def convert_to_cupy(x):
    if isinstance(x, torch.Tensor):
        return xp.array(x.detach().cpu().numpy())
    elif isinstance(x, numpy.ndarray):
        return xp.array(x)
    else:
        return x

def convert_to_torch(x):
    if isinstance(x, xp.ndarray):
        return torch.from_numpy(x.get())
    elif isinstance(x, numpy.ndarray):
        return torch.from_numpy(x)
    else:
        return x
    
def convert_to_numpy(x):
    if isinstance(x, xp.ndarray):
        return x.get()
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


from torchkbnufft import KbNufft
class NUFFT(torch.nn.Module):
    def __init__(self, pixel_position_reciprocal, over_sampling=1, device='cpu'):
        super().__init__()
        self.register_buffer("pixel_position_reciprocal", pixel_position_reciprocal)
        self.over_sampling = over_sampling
        self.im_size = pixel_position_reciprocal.shape[1]
        self.nufft_forward = KbNufft(im_size=(self.im_size,)*3)
        self.device = device
        self.to(device)
        
    def forward(self, ac, quaternions, compute_grad=False):
        if compute_grad:
            return self._forward_with_grad(ac, quaternions)
        else:
            with torch.no_grad():
                return self._forward_with_grad(ac, quaternions)
    
    def _forward_with_grad(self, ac, quaternions):
        ac = convert_to_torch(ac).to(self.device)
        quaternions = convert_to_torch(quaternions).to(self.device)

        HKL = gen_nonuniform_normalized_positions(
                quaternions, self.pixel_position_reciprocal, self.over_sampling)
        if not ac.is_complex():
            ac = ac.to(dtype=torch.complex64)
        if ac.ndim == 3:
            ac = ac.unsqueeze(0).unsqueeze(0)
        assert ac.ndim == 5, f"ac must be 5D tensor, got {ac.ndim}"
        slices = self.nufft_forward(ac, HKL)[0,0]
        slices = slices.real.view((-1,) + (self.im_size,)*2)
        return slices

"""
taken from https://scikit-surgerycore.readthedocs.io/en/stable/_modules/sksurgerycore/algorithms/averagequaternions.html#average_quaternions
"""

def nice_orientation(quaternions, weights=None):
    if weights is None:
        weights = torch.ones(quaternions.shape[0]).to(quaternions)
    weighted_mean = torch.sum(weights.view(-1, 1) * quaternions, dim=0)
    dp = torch.einsum('j,ij->i', weighted_mean, quaternions)
    sign = torch.sign(dp)
    corrected_quaternions = quaternions.clone()
    corrected_quaternions[sign < 0] = -quaternions[sign < 0]
    return corrected_quaternions

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

    quaternions = nice_orientation(quaternions)
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
    quaternions = nice_orientation(quaternions)
    # Number of quaternions to average
    samples = quaternions.shape[0]
    
    mat_a = torch.einsum('lk, ki, kj -> lij', weights, quaternions, quaternions)
    weight_sum = weights.sum(dim=1)

    if torch.any(weight_sum <= 0):
        raise ValueError("At least one weight must be greater than zero")

    # scale
    mat_a = (1.0 / weight_sum.view(-1,1,1)) * mat_a

    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = torch.linalg.eig(mat_a)

    # Sort by largest eigenvalue
    max_eigen_vectors = eigen_vectors[torch.arange(eigen_vectors.shape[0]).to(eigen_vectors.device), :, eigen_values.real.argsort(descending=True, dim=-1)[:,0]]

    # return the real part of the largest eigenvector (has only real part)
    return max_eigen_vectors.real

def weighted_average_matrices(matrices, weights):
    quaternions = matrix_to_quaternion(matrices)
    avg_quaternion = weighted_average_quaternions(quaternions, weights)
    avg_matrix = quaternion_to_matrix(avg_quaternion)
    return avg_matrix

class TestQuaternionFunctions(unittest.TestCase):
    
    def test_weighted_average(self):
        for i in range(10):
            base_quat = skopi.get_random_quat(1)
            print(base_quat)
            quats = skopi.get_preferred_orientation_quat(sigma=2, num_pts=10000, base_quat=base_quat[0])
            avg_mat = weighted_average_matrices(
                quaternion_to_matrix(torch.from_numpy(quats)), 
                torch.ones(1, len(quats)) / len(quats)
            )
            
            avg_quat = matrix_to_quaternion(avg_mat).numpy()
            print(avg_quat)
            if np.sign(np.dot(base_quat[0], avg_quat[0])) < 0:
                avg_quat = -avg_quat
            self.assertTrue(np.linalg.norm(base_quat - avg_quat) < 1e-1,
                            msg=f"Quaternions not close enough. Base: {base_quat}, Avg: {avg_quat}, Diff: {np.linalg.norm(base_quat - avg_quat)}")

            # Add assertions here if you have criteria to check the results
            # For example:
            # self.assertGreater(some_value, avg_quat, "Value is not as expected")
            
if __name__ == "__main__":
    import skopi
    import numpy as np
    unittest.main()
