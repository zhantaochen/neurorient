
try:
    import cupy as xp
    from cupyx.scipy.ndimage import gaussian_filter
    use_cupy = True
except ImportError:
    import numpy as xp
    from scipy.ndimage import gaussian_filter
    use_cupy = False


import matplotlib.pyplot as plt
import numpy             as np

from scipy.ndimage       import gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg

import skopi as skp

import torch
import torchkbnufft as tkbn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def nufft_3d_t1(H, K, L, c, M):
    # kb = tkbn.KbNufft(im_size=(M, M, M))
    kb_adj = tkbn.KbNufftAdjoint(im_size=(M, M, M)).to(device)
    omega = np.vstack([H.flatten()[None],
                       K.flatten()[None],
                       L.flatten()[None]])
    omega = torch.from_numpy(omega).to(torch.float32).to(device)
    c = torch.from_numpy(c[None,None]).to(torch.complex64).to(device)

    out = kb_adj(c, omega).detach().cpu().numpy()
    return out
    


def adjoint(nuvect, H_, K_, L_, support, M, recip_extent, use_recip_sym):
    """Apply the adjoint, NUFFT1+ problem"""

    # Ensure that H_, K_, and L_ have the same shape
    assert H_.shape == K_.shape == L_.shape

    # Solve the NUFFT
    ugrid = nufft_3d_t1(H_, K_, L_, nuvect, M)

    # Apply support
    ugrid *= support

    # Apply recip symmetry
    if use_recip_sym:
        ugrid = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(ugrid.reshape((M,)*3))).real)).real
        ##### FIXME: ugrid = ugrid.real -> do fft stuff above

    return ugrid / M**3

def fourier_reg(uvect, support, F_antisupport, M, use_recip_sym):
    ugrid = uvect.reshape((M,) * 3) * support
    if use_recip_sym:
        assert np.all(np.isreal(ugrid))

    if use_cupy:
        uvect = xp.asarray(uvect)
        support = xp.asarray(support)
        F_antisupport = xp.asarray(F_antisupport)

    F_ugrid = xp.fft.fftn(xp.fft.ifftshift(ugrid)) #/ M**3
    F_reg = F_ugrid * xp.fft.ifftshift(F_antisupport)
    reg = xp.fft.fftshift(xp.fft.ifftn(F_reg))
    uvect = (reg * support).flatten()
    if use_recip_sym:
        uvect = uvect.real

    if use_cupy:
        uvect = xp.asnumpy(uvect)
    return uvect

def gen_nonuniform_positions(orientations, pixel_position_reciprocal):
    # Generate q points (h,k,l) from the given rotations and pixel positions 

    if orientations.shape[0] > 0:
        rotmat = np.array([np.linalg.inv(skp.quaternion2rot3d(quat)) for quat in orientations])
    else:
        rotmat = np.zeros((0, 3, 3))
        print("WARNING: gen_nonuniform_positions got empty orientation - returning h,k,l for Null rotation")

    # TODO: How to ensure we support all formats of pixel_position reciprocal
    # Current support shape is (3, N_panels, Dim_x, Dim_y) 
    H, K, L = np.einsum("ijk,lmnk->jilmn", rotmat, pixel_position_reciprocal)
    #H, K, L = np.einsum("ijk,klm->jilm", rotmat, pixel_position_reciprocal)
    # shape -> [N_images] x det_shape
    return H, K, L

def core_problem_convolution(uvect, M, F_ugrid_conv_, M_ups, ac_support,
                             use_reciprocal_symmetry):
    if use_cupy:
        uvect = xp.asarray(uvect)
        ac_support = xp.asarray(ac_support)
        F_ugrid_conv_ = xp.asarray(F_ugrid_conv_)
    
    # Upsample
    uvect = xp.fft.fftshift(xp.fft.ifftn(xp.fft.fftn(xp.fft.ifftshift(uvect.reshape((M,)*3))).real)).real ##### FIXME: np->xp & fftshift->ifftshift & ifftshift->fftshift
    print(f"##### core_problem uvect: {np.where(uvect==np.max(uvect))}")
    ugrid = uvect * ac_support
    ugrid_ups = xp.zeros((M_ups,) * 3, dtype=uvect.dtype)            
    ugrid_ups[:M, :M, :M] = ugrid
    
    # Convolution = Fourier multiplication
    F_ugrid_ups = xp.fft.fftn(xp.fft.ifftshift(ugrid_ups)) / M**3 * (M_ups/M)**3 ##### FIXME: ifftshift -> fftshift
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = xp.fft.fftshift(xp.fft.ifftn(F_ugrid_conv_out_ups)) ##### FIXME: fftshift -> ifftshift
    
    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]
    ugrid_conv_out *= ac_support

    # Apply recip symmetry
    if use_reciprocal_symmetry:
        # Both ugrid_conv and ugrid are real, so their convolution
        # should be real, but numerical errors accumulate in the
        # imaginary part.
        ugrid_conv_out = xp.fft.fftshift(xp.fft.ifftn(xp.fft.fftn(xp.fft.ifftshift(ugrid_conv_out.reshape((M,)*3))).real)).real ##### FIXME: np->xp & fftshift->ifftshift & ifftshift->fftshift & no need to reshape ugrid_conv_out which is already (M,M,M)

    if use_cupy:
        ugrid_conv_out = xp.asnumpy(ugrid_conv_out)
    return ugrid_conv_out.flatten()


def setup_linops(H, K, L, data,
                 ac_support, weights, x0,
                 M, N, reciprocal_extent,
                 rlambda, flambda,
                 use_reciprocal_symmetry, oversampling=1):
    """Define W and d parts of the W @ x = d problem.

    W = A_adj*Da*A + rl*I  + fl*F_adj*Df*F
    d = A_adj*Da*b + rl*x0 + 0

    Where:
        A represents the NUFFT operator
        A_adj its adjoint
        I the identity
        F the FFT operator
        F_adj its atjoint
        Da, Df weights
        b the data
        x0 the initial guess (ac_estimate)
    """
    H_ = H.flatten() / reciprocal_extent * np.pi / oversampling
    K_ = K.flatten() / reciprocal_extent * np.pi / oversampling
    L_ = L.flatten() / reciprocal_extent * np.pi / oversampling

    lu = np.linspace(-np.pi, np.pi, M)
    Hu_, Ku_, Lu_ = np.meshgrid(lu, lu, lu, indexing='ij')
    Qu_ = np.sqrt(Hu_**2 + Ku_**2 + Lu_**2)
    F_antisupport = Qu_ > np.pi / oversampling
    assert np.all(F_antisupport == F_antisupport[::-1, :, :])
    assert np.all(F_antisupport == F_antisupport[:, ::-1, :])
    assert np.all(F_antisupport == F_antisupport[:, :, ::-1])
    assert np.all(F_antisupport == F_antisupport[::-1, ::-1, ::-1])

    # Using upsampled convolution technique instead of ADA
    M_ups = M * oversampling
    ugrid_conv = adjoint(
        np.ones_like(data), H_, K_, L_, 1, M_ups,
        reciprocal_extent, use_reciprocal_symmetry)
    F_ugrid_conv_ = np.fft.fftn(np.fft.ifftshift(ugrid_conv)) / M**3

    def W_matvec(uvect):
        """Define W part of the W @ x = d problem."""
        uvect_ADA = core_problem_convolution(
            uvect, M, F_ugrid_conv_, M_ups, ac_support, use_reciprocal_symmetry)
        uvect_FDF = fourier_reg(
            uvect, ac_support, F_antisupport, M, use_reciprocal_symmetry)
        uvect = uvect_ADA + rlambda*uvect + flambda*uvect_FDF
        return uvect

    W = LinearOperator(
        dtype=np.complex64,
        shape=(M**3, M**3),
        matvec=W_matvec)

    nuvect_Db = data * weights
    uvect_ADb = adjoint(
        nuvect_Db, H_, K_, L_, ac_support, M,
        reciprocal_extent, use_reciprocal_symmetry
    ).flatten()
    d = uvect_ADb + rlambda*x0

    return W, d


def solve_ac(pixel_position_reciprocal,
             pixel_distance_reciprocal,
             slices_,
             maxiter=100,
             orientations=None,
             ac_estimate=None):
    M = slices_.shape[-1]
    N_images = slices_.shape[0]
    N = np.prod(slices_.shape)
    reciprocal_extent = pixel_distance_reciprocal.max()
    use_reciprocal_symmetry = True

    if orientations is None:
        orientations = skp.get_random_quat(N_images)
    H, K, L = gen_nonuniform_positions(
        orientations, pixel_position_reciprocal)

    if ac_estimate is None:
        ac_support = np.ones((M,)*3)
        ac_estimate = np.zeros((M,)*3)
    else:
        ac_smoothed = gaussian_filter(ac_estimate, 0.5)
        ac_support = (ac_smoothed > 1e-12).astype(np.float)
        ac_estimate *= ac_support
    
    data = slices_.flatten()
    weights = np.ones(N)

    # regularization paramters
    rlambda = 1/N / 1000
    flambda = 1e3

    x0 = ac_estimate.flatten()
    W, d = setup_linops(H, K, L, data,
                        ac_support, weights, x0,
                        M, N, reciprocal_extent,
                        rlambda, flambda,
                        use_reciprocal_symmetry)
    ret, info = cg(W, d, x0=x0, maxiter=maxiter)
    ac = ret.reshape((M,)*3)
    if use_reciprocal_symmetry:
        assert np.all(np.isreal(ac))
    ac = ac.real

    return ac
