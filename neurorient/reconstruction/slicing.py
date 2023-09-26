# taken from https://gitlab.osti.gov/mtip/spinifel/

from math import pi
import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

from scipy.interpolate import RegularGridInterpolator

def get_rho_function(real_mesh, rho):
    if isinstance(real_mesh, torch.Tensor):
        real_mesh = real_mesh.detach().cpu().numpy()
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()
    real_mesh = np.around(real_mesh, 3)
    x = np.sort(np.unique(real_mesh[...,0]))
    y = np.sort(np.unique(real_mesh[...,1]))
    z = np.sort(np.unique(real_mesh[...,2]))
    assert len(x) == rho.shape[0]
    assert len(y) == rho.shape[1]
    assert len(z) == rho.shape[2]
    rho_func = RegularGridInterpolator((x, y, z), rho, bounds_error=False, fill_value=0.)
    return rho_func

def get_reciprocal_mesh(voxel_number_1d, max_reciprocal_value):
    """
    Get a centered, symetric mesh of given dimensions. Altered from skopi.

    Parameters
    ----------
    voxel_number_1d : int
        number of voxels per axis
    max_reciprocal_value : float
        maximum voxel resolution in inverse Angstrom
    xp: package
        use numpy or cupy

    Returns
    -------
    reciprocal_mesh : numpy.ndarray, shape (n,n,n,3)
        grid of reciprocal space vectors for each voxel
    """
    linspace = torch.linspace(-max_reciprocal_value, max_reciprocal_value, voxel_number_1d)
    reciprocal_mesh_stack = torch.stack(
        torch.meshgrid(linspace, linspace, linspace, indexing='ij'))
    reciprocal_mesh = torch.moveaxis(reciprocal_mesh_stack, 0, -1)

    return reciprocal_mesh

def reciprocal_mesh_2_real_mesh(reciprocal_mesh):
    _lin = torch.linspace(-reciprocal_mesh.max(), reciprocal_mesh.max(), reciprocal_mesh.shape[0])
    step = _lin[1] - _lin[0]
    max_real_value = 1 / (2*step)
    linspace = torch.linspace(-max_real_value, max_real_value, reciprocal_mesh.shape[0]) * 1e10
    real_mesh_stack = torch.stack(
            torch.meshgrid(linspace, linspace, linspace, indexing='ij'))
    real_mesh = torch.moveaxis(real_mesh_stack, 0, -1)
    return real_mesh

def real_mesh_2_reciprocal_mesh(real_mesh):
    _lin = torch.linspace(-real_mesh.max(), real_mesh.max(), real_mesh.shape[0])
    step = _lin[1] - _lin[0]
    max_reciprocal_value = 1 / (2*step)
    linspace = torch.linspace(-max_reciprocal_value, max_reciprocal_value, real_mesh.shape[0]) * 1e10
    reciprocal_mesh_stack = torch.stack(
            torch.meshgrid(linspace, linspace, linspace, indexing='ij'))
    reciprocal_mesh = torch.moveaxis(reciprocal_mesh_stack, 0, -1)
    return reciprocal_mesh
    
def get_real_mesh(voxel_number_1d, max_reciprocal_value, return_reciprocal=False):
    """
    
    Parameters
    ----------
    voxel_number_1d : int
        number of voxels per axis
    max_reciprocal_value : float
        maximum voxel resolution in inverse Angstrom

    Returns
    -------
    real_mesh : numpy.ndarray, shape (n,n,n,3)
        grid of real space vectors for each voxel
    """
    
    reciprocal_mesh = get_reciprocal_mesh(voxel_number_1d, max_reciprocal_value)
    _lin = torch.linspace(-reciprocal_mesh.max(), reciprocal_mesh.max(), voxel_number_1d)
    step = _lin[1] - _lin[0]
    max_real_value = 1 / (2*step)
    linspace = torch.linspace(-max_real_value, max_real_value, voxel_number_1d) * 1e10
    real_mesh_stack = torch.stack(
            torch.meshgrid(linspace, linspace, linspace, indexing='ij'))
    real_mesh = torch.moveaxis(real_mesh_stack, 0, -1)
    if return_reciprocal:
        return real_mesh, reciprocal_mesh
    else:
        return real_mesh

# def get_real_mesh(voxel_number_1d, max_reciprocal_value):
#     """
    
#     Parameters
#     ----------
#     voxel_number_1d : int
#         number of voxels per axis
#     max_reciprocal_value : float
#         maximum voxel resolution in inverse Angstrom

#     Returns
#     -------
#     real_mesh : numpy.ndarray, shape (n,n,n,3)
#         grid of real space vectors for each voxel
#     """
    
#     _lin = torch.linspace(-max_reciprocal_value, max_reciprocal_value, voxel_number_1d)
#     step = _lin[1] - _lin[0]
#     max_real_value = 1 / (2*step)
#     linspace = torch.linspace(-max_real_value, max_real_value, voxel_number_1d) * 1e10
#     real_mesh_stack = torch.stack(
#             torch.meshgrid(linspace, linspace, linspace, indexing='ij'))
#     real_mesh = torch.moveaxis(real_mesh_stack, 0, -1)
    
#     return real_mesh

def gen_nonuniform_positions(orientations, pixel_position_reciprocal):
    # Generate q points (h,k,l) from the given rotations and pixel positions

    if orientations.shape[0] > 0:
        if orientations.ndim == 2:
            rotmat = torch.linalg.inv(quaternion_to_matrix(orientations))
        elif orientations.ndim == 3:
            rotmat = torch.linalg.inv(orientations)
    else:
        rotmat = torch.zeros((0, 3, 3)).to(orientations)
        # logger.log(
        #     "WARNING: gen_nonuniform_positions got empty orientation - returning h,k,l for Null rotation",
        #     level=1
        # )

    # TODO: How to ensure we support all formats of pixel_position reciprocal
    # H, K, L = torch.einsum("ijk,klmn->jilmn", rotmat, pixel_position_reciprocal)
    # Current support shape is (N_panels, Dim_x, Dim_y, 3)
    # Einsum shape is (3, N_images, ) + det_shape
    # H, K, L shape -> [N_images, ] + det_shape
    HKL = torch.einsum("ijk,lmnk->jilmn", rotmat, pixel_position_reciprocal)
    # shape -> [H,K,L] x [N_images] x det_shape
    return HKL


def gen_nonuniform_normalized_positions(
    orientations, pixel_position_reciprocal, oversampling
):
    HKL = gen_nonuniform_positions(orientations, pixel_position_reciprocal)

    # TODO: Control/set precisions needed here
    # scale and change type for compatibility with finufft
    # HKL = HKL.view(3, -1) / pixel_position_reciprocal.norm(dim=-1).max() * pi / oversampling
    HKL = HKL.view(3, -1) / pixel_position_reciprocal.max() * pi / oversampling
    return HKL


def gen_model_slices(nufft_forward, ac, ref_orientations,
                     pixel_position_reciprocal,
                     oversampling=1
                     ):
    """
    Generate model slices using given reference orientations (in quaternion)
    """

    # Get q points (normalized by recirocal extent and oversampling)
    HKL = gen_nonuniform_normalized_positions(
        ref_orientations, pixel_position_reciprocal, oversampling)

    # nuvect = autocorrelation.forward(
    #     ac, H_, K_, L_, 1, ac_support_size, N, reciprocal_extent, True)
    
    ac = ac.real + 0j
    nuvect = nufft_forward(ac.unsqueeze(0).unsqueeze(0), HKL)[0,0]

    model_slices = nuvect.real

    return model_slices
