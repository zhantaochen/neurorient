# adapted from pytorch3d.transformation functions

import torch
import torch.nn.functional as F
from pytorch3d.transforms.rotation_conversions import random_rotations
from pytorch3d.transforms import acos_linear_extrapolation

def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    dim1, dim2 = R.shape[-2:]
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)
        
def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-2,
) -> torch.Tensor:
    """
    Calculates the relative angle between each pair of rotation matrices from `R1` and `R2`.
    Returns a tensor of shape (m, n) containing the angles.
    """
    if R1.dim() != 3 or R2.dim() != 3 or R1.size(1) != 3 or R1.size(2) != 3 or R2.size(1) != 3 or R2.size(2) != 3:
        raise ValueError("Both R1 and R2 must be of shape (*, 3, 3)")
    
    # Batch matrix multiplication with broadcasting
    R1_exp = R1.unsqueeze(1)  # Shape (m, 1, 3, 3)
    R2_exp = R2.unsqueeze(0)  # Shape (1, n, 3, 3)
    R12 = torch.matmul(R1_exp, R2_exp.transpose(-2, -1))  # Shape (m, n, 3, 3)
    shape = R12.shape[:-2]
    
    R12 = R12.reshape(-1, 3, 3)
    """handle improper rotations
    """
    R12_det = torch.linalg.det(R12)
    R12[R12_det < 0] = -R12[R12_det < 0]
    
    # Compute angles
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps).reshape(shape)
