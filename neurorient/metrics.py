from cupyx.scipy.interpolate import RegularGridInterpolator

import numpy as np
import cupy as cp

import scipy
import torch

from .utils_transform import convert_to_cupy, convert_to_torch
from .reconstruction.slicing import real_mesh_2_reciprocal_mesh
from .external.spinifel import align_volumes

def compute_fsc(
        volume1,
        mesh1,
        volume2,
        mesh2=None,
        volume_type='electron_density',
        q_spacing=0.01):
    """
    Taken from https://gitlab.osti.gov/mtip/spinifel/-/blob/master/eval/fsc.py?ref_type=heads

    Compute the Fourier shell correlation (FSC) curve, with the
    estimated resolution based on a threshold of 0.5.

    Parameters
    ----------
    volume1 : numpy.ndarray, shape (n,n,n)
        reference map
    volume2 : numpy.ndarray, shape (n,n,n)
        reconstructed map
    distance_reciprocal_max : float
        maximum voxel resolution in inverse Angstrom
    q_spacing : float
        q_spacing for evaluating FSC in inverse Angstrom

    Returns
    -------
    resolution : float
        estimated resolution of reconstructed map in Angstroms
    """

    volume1 = convert_to_cupy(volume1)
    volume2 = convert_to_cupy(volume2)
    mesh1 = convert_to_cupy(mesh1)

    if mesh2 is None:
        assert volume1.shape == volume2.shape, "Volumes must be the same shape if mesh2 is not provided."
        mesh2 = mesh1
    else:
        mesh2 = convert_to_cupy(mesh2)
        if mesh2.max() > mesh1.max():
            func_vol2 = RegularGridInterpolator((mesh2[:, 0, 0, 0], mesh2[0, :, 0, 1], mesh2[0, 0, :, 2]), volume2, bounds_error=False, fill_value=0.)
            volume2 = func_vol2(mesh1)
            mesh_major = mesh1
        else:
            func_vol1 = RegularGridInterpolator((mesh1[:, 0, 0, 0], mesh1[0, :, 0, 1], mesh1[0, 0, :, 2]), volume1, bounds_error=False, fill_value=0.)
            volume1 = func_vol1(mesh2)
            mesh_major = mesh2

    # align volumes
    volume2, volume1, aligned_cc = align_volumes(volume2, volume1, zoom=0.3)
    volume2 = convert_to_cupy(volume2)

    if volume_type == 'electron_density':
        mesh = convert_to_cupy(real_mesh_2_reciprocal_mesh(convert_to_torch(mesh_major)))
        ft1 = cp.fft.fftshift(cp.fft.fftn(volume1)).reshape(-1)
        ft2 = cp.conjugate(cp.fft.fftshift(cp.fft.fftn(volume2)).reshape(-1))
    elif volume_type == 'intensity':
        mesh = mesh_major
        ft1 = cp.sqrt(volume1.clip(0.)).reshape(-1)
        ft2 = cp.sqrt(volume2.clip(0.)).reshape(-1)
    
    q_spacing = min(2.5e-10*(mesh[1,0,0,0] - mesh[0,0,0,0]).get(), q_spacing)

    smags = cp.linalg.norm(cp.array(mesh), axis=-1).reshape(-1) * 1e-10
    q_bounds = cp.arange(0, smags.max() / cp.sqrt(3), q_spacing)
    q_centers = (q_bounds[:-1] + q_bounds[1:]) / 2

    fsc = cp.zeros(len(q_bounds)-1)

    for i, r in enumerate(q_bounds[:-1]):
        indices = cp.where((smags > r) & (smags < r + q_spacing))[0]
        numerator = cp.sum(ft1[indices] * ft2[indices])
        denominator = cp.sqrt(
            cp.sum(
                cp.square(
                    cp.abs(
                        ft1[indices]))) *
            cp.sum(
                    cp.square(
                        cp.abs(
                            ft2[indices]))))
        fsc[i] = numerator.real / (denominator + 1e-12)

    if not isinstance(fsc, np.ndarray):
        fsc = fsc.get()
        q_centers = q_centers.get()

    f = scipy.interpolate.interp1d(fsc, q_centers)
    try:
        resolution = 1.0 / f(0.5)
        print(f"Estimated resolution from FSC: {resolution:.1f} Angstrom")
    except ValueError:
        if fsc.min() > 0.5:
            resolution = 1.0 / q_centers[-1]
            print(f"Estimated resolution from largest-q: at least {resolution:.1f} Angstrom")
        else:
            resolution = -1
            print("Resolution could not be estimated.")

    return resolution, q_centers, fsc

