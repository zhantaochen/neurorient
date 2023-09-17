import numpy as np
import mrcfile
from .geom import *
import skopi as sk
from .config import xp, ndimage

def rotate_volume(vol, quat):
    """
    Rotate copies of the volume by the given quaternions.

    Parameters
    ----------
    vol : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    quat : numpy.ndarray, shape (n_quat,4)
        quaternions to apply to the volume

    Returns
    -------
    rot_vol : numpy.ndarray, shape (n_quat,n,n,n)
        rotated copies of volume
    """
    vol = xp.array(vol)
    quat = xp.array(quat)

    M = vol.shape[0]
    lincoords = xp.arange(M)
    coords = xp.meshgrid(lincoords, lincoords, lincoords)

    xyz = xp.vstack([coords[0].reshape(-1) - int(M / 2),
                     coords[1].reshape(-1) - int(M / 2),
                     coords[2].reshape(-1) - int(M / 2)])

    R = quaternion2rot3d(quat)
    transformed_xyz = xp.dot(R, xyz) + int(M / 2)

    new_xyz = xp.array([transformed_xyz[:, 1, :].flatten(),
                        transformed_xyz[:, 0, :].flatten(),
                        transformed_xyz[:, 2, :].flatten()])
    rot_vol = ndimage.map_coordinates(vol, new_xyz, order=1)
    rot_vol = rot_vol.reshape((quat.shape[0], M, M, M))
    return rot_vol

def center_volume(vol):
    """
    Apply translational shifts to center the density within the volume.

    Parameters
    ----------
    vol : numpy.ndarray, shape (n,n,n)
        volume to be centered

    Returns
    -------
    cen_vol : numpy.ndarray, shape (n,n,n)
        centered volume
    """
    vol = xp.array(vol)
    old_center = xp.array(ndimage.center_of_mass(vol))
    new_center = xp.array(xp.array(vol.shape) / 2).astype(int)
    cen_vol = ndimage.shift(vol, -1 * (old_center - new_center), order=0)
    return cen_vol

def pearson_cc(arr1, arr2):
    """
    Compute the Pearson correlation-coefficient between the input arrays.

    Parameters
    ----------
    arr1 : numpy.ndarray, shape (n_samples, n_points)
        input array
    arr2 : numpy.ndarray, shape (n_samples, n_points) or (1, n_points)
        input array to compute CC with

    Returns
    -------
    ccs : numpy.ndarray, shape (n_samples)
        correlation coefficient between paired sample arrays, or if
        arr2.shape[0] == 1, then between each sample of arr1 to arr2
    """
    vx = arr1 - arr1.mean(axis=-1)[:, None]
    vy = arr2 - arr2.mean(axis=-1)[:, None]
    numerator = xp.sum(vx * vy, axis=1)
    # Merge Square root because we have squares
    denom = xp.sqrt(xp.sum(vx**2, axis=1)) * xp.sqrt(xp.sum(vy**2, axis=1))
    return numerator / denom

def score_deformations(mrc1, mrc2, warp):
    """
    Compute the Pearson correlation coefficient between the input volumes
    after rotating or displacing the first volume by the given quaternions
    or translations.

    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be warped
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    warp : numpy.ndarray, shape (n_quat,4) or (n_trans,3)
        rotations or displacements to apply to mrc2

    Returns
    -------
    ccs : numpy.ndarray, shape (n_quat)
        correlation coefficients associated with warp
    """
    # deform one of the input volumes by rotation or displacement
    if warp.shape[-1] == 4:
        wmrc1 = rotate_volume(mrc1, warp)
    elif warp.shape[-1] == 3:
        print("Not yet implemented")
        return
    else:
        print("Warp input must be quaternions or translations")
        return

    # score each deformed volume using the Pearson CC
    wmrc1_flat = wmrc1.reshape(wmrc1.shape[0], -1)
    mrc2_flat = xp.expand_dims(mrc2.flatten(), axis=0)
    ccs = pearson_cc(wmrc1_flat, mrc2_flat)
    return ccs

def scan_orientations_fine(
        mrc1,
        mrc2,
        opt_q,
        prev_score,
        n_iterations=10,
        n_search=420):
    """
    Perform a fine alignment search in the vicinity of the input quaternion
    to align mrc1 to mrc2.

    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    opt_q : numpy.ndarray, shape (1,4)
        starting quaternion to apply to mrc1 to align it with mrc2
    prev_score : float
        cross-correlation associated with alignment quat opt_q
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation

    Returns
    -------
    opt_q : numpy.ndarray, shape (4)
        quaternion to apply to mrc1 to align it with mrc2
    prev_score : float
        cross-correlation between aligned mrc1 and mrc2
    """
    # perform a series of fine alignment, ending if CC no longer improves
    sigmas = 2 - 0.2 * xp.arange(1, 10)
    for n in range(1, n_iterations):
        quat = get_preferred_orientation_quat(
            n_search - 1, float(sigmas[n - 1]), base_quat=opt_q)
        quat = xp.vstack((opt_q, quat))
        ccs = score_deformations(mrc1, mrc2, quat)
        if xp.max(ccs) < prev_score:
            break
        else:
            opt_q = quat[xp.argmax(ccs)]
        #print(torch.max(ccs), opt_q) # useful for debugging
        prev_score = xp.max(ccs)

    return opt_q, prev_score

def scan_orientations(
        mrc1,
        mrc2,
        n_iterations=10,
        n_search=420,
        nscs=1):
    """
    Find the quaternion and its associated score that best aligns volume mrc1 to mrc2.
    Candidate orientations are scored based on the Pearson correlation coefficient.
    First a coarse search is performed, followed by a series of increasingly fine
    searches in angular space. To prevent getting stuck in a bad solution, the top nscs
    solutions from the coarse grained search can be investigated.

    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation
    nscs : int, default 1
        number of solutions from the coarse-grained search to investigate

    Returns
    -------
    opt_q : numpy.ndarray, shape (4)
        quaternion to apply to mrc1 to align it with mrc2
    score : float
        cross-correlation between aligned mrc1 and mrc2
    """
    # perform a coarse alignment to start
    quat = xp.array(sk.get_uniform_quat(n_search)) # update for skopi
    ccs = score_deformations(mrc1, mrc2, quat)
    ccs_order = xp.argsort(ccs)[::-1]

    # scan the top solutions
    opt_q_list, ccs_list = xp.zeros((nscs, 4)), xp.zeros(nscs)
    for n in range(nscs):
        start_q, start_score = quat[ccs_order[n]], ccs[ccs_order[n]]
        opt_q_list[n], ccs_list[n] = scan_orientations_fine(
            mrc1, mrc2, start_q, start_score, n_iterations=n_iterations, n_search=n_search)

    opt_q, score = opt_q_list[xp.argmax(ccs_list)], xp.max(ccs_list)
    return opt_q, score

def align_volumes(
        mrc1,
        mrc2,
        zoom=1,
        sigma=0,
        n_iterations=10,
        n_search=420,
        nscs=1,):
    """
    Find the quaternion that best aligns volume mrc1 to mrc2. Volumes are
    optionally preprocessed by up / downsampling and applying a Gaussian
    filter.

    Parameters
    ----------
    mrc1 : numpy.ndarray, shape (n,n,n)
        volume to be rotated
    mrc2 : numpy.ndarray, shape (n,n,n)
        volume to be held fixed
    zoom : float, default 1
        if not 1, sample by which to up or downsample volume
    sigma : int, default 0
        sigma of Gaussian filter to apply to each volume
    n_iterations: int, default 10
        number of iterations of alignment to perform
    n_search : int, default 420
        number of quaternions to score at each orientation
    nscs : int, default 1
        number of solutions from the coarse-grained alignment search to investigate
    output : string, default None
        if supplied, save the aligned volume to this path in MRC format
    voxel_size : float, default None
        if supplied, use as value of voxel size in Angstrom for output

    Returns
    -------
    r_vol : numpy.ndarray, shape (n,n,n)
        copy of centered mrc1 aligned with centered mrc2
    mrc2_original : umpy.ndarray, shape (n,n,n)
        copy of centered mrc2
    """
    # center input volumes, then make copies
    mrc1, mrc2 = xp.array(mrc1), xp.array(mrc2)
    mrc1, mrc2 = center_volume(mrc1), center_volume(mrc2)
    mrc1_original = mrc1.copy()
    mrc2_original = mrc2.copy()
    
    init_cc = pearson_cc(
        xp.expand_dims(mrc1_original.flatten(), axis=0), 
        xp.expand_dims(mrc2_original.flatten(), axis=0))[0]
    print(
        f"Initial CC between unzoomed / unfiltered volumes is: {init_cc:.3f}")
    
    # optionally up/downsample volumes
    if zoom != 1:
        mrc1 = ndimage.zoom(mrc1, (zoom, zoom, zoom))
        mrc2 = ndimage.zoom(mrc2, (zoom, zoom, zoom))

    # optionally apply a Gaussian filter to volumes
    if sigma != 0:
        mrc1 = ndimage.gaussian_filter(mrc1, sigma=sigma)
        mrc2 = ndimage.gaussian_filter(mrc2, sigma=sigma)

    # evaluate both hands
    opt_q1, cc1 = scan_orientations(
        mrc1, mrc2, n_iterations, n_search, nscs=nscs)
    opt_q2, cc2 = scan_orientations(
        flip(mrc1, [0, 1, 2]), mrc2, n_iterations, n_search, nscs=nscs)
    if cc1 > cc2:
        opt_q, cc_r, invert = opt_q1, cc1, False
    else:
        opt_q, cc_r, invert = opt_q2, cc2, True
    print(f"Alignment CC after rotation is: {cc_r:.3f}")

    # generate final aligned map
    if invert:
        print("Map had to be inverted")
        mrc1_original = flip(mrc1_original, [0, 1, 2])
    r_vol = rotate_volume(mrc1_original, xp.expand_dims(opt_q, axis=0))[0]
    final_cc = pearson_cc(
        xp.expand_dims(r_vol.flatten(), axis=0), 
        xp.expand_dims(mrc2_original.flatten(), axis=0))[0]
    print(
        f"Final CC between unzoomed / unfiltered volumes is: {final_cc:.3f}")

    if init_cc >= final_cc:
        print("Warning: CC decreased after alignment, returning original volume.")
        return mrc1_original, mrc2_original, init_cc
    else:
        return r_vol, mrc2_original, final_cc

# Work around for bug in cupy flip and its normalize axis indicies
def flip(arr, orien):
    if not isinstance(arr, np.ndarray):
        return xp.array(np.flip(arr.get(), orien))
    else:
        return np.flip(arr, orien)
