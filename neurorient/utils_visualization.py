import matplotlib.pyplot as plt
import numpy as np
import torch

import mrcfile
from .utils_transform import convert_to_numpy

def save_mrc(output, data, voxel_size=None, header_origin=None):
    """
    taken from Spinifel
    
    Save numpy array as an MRC file.

    Parameters
    ----------
    output : string, default None
        if supplied, save the aligned volume to this path in MRC format
    data : numpy.ndarray
        image or volume to save
    voxel_size : float, default None
        if supplied, use as value of voxel size in Angstrom in the header
    header_origin : numpy.recarray
        if supplied, use the origin from this header object
    """
    mrc = mrcfile.new(output, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    if header_origin is not None:
        mrc.header['origin']['x'] = float(header_origin['origin']['x'])
        mrc.header['origin']['y'] = float(header_origin['origin']['y'])
        mrc.header['origin']['z'] = float(header_origin['origin']['z'])
        mrc.update_header_from_data()
        mrc.update_header_stats()
    mrc.close()
    return

def display_fsc(q, fsc, resolution=None, criterion=0.5, save_to=None, closefig=False):
    fig, ax1 = plt.subplots()
    ax1.plot(q, fsc)
    ax1.set_xticks(np.linspace(0, np.round(q.max(), decimals=2), 5))
    ax1.set_xlabel('Reciprocal distance $q$ ($\mathrm{\AA}^{-1}$)', fontsize=14)
    ax1.set_ylabel('Fourier Shell Correlation (FSC)', fontsize=14)

    if resolution is not None:
        ax1.hlines(criterion, -0.1, 1 / resolution, linestyles='--', colors='k')
        ax1.vlines(1 / resolution, criterion, ax1.get_ylim()[1], linestyles='--', colors='k')
        ax1.text(1 / resolution + 0.005, 
                 criterion + 0.05, 
                 f'Resolution: {resolution:.2f} $\mathrm{{\AA}}$', fontsize=11, ha='left')
    ax1.set_xlim([-0.005, q.max()+0.005])
    ax1.set_ylim([min(-0.05, fsc.min()-0.025), 1.05])

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels([f"{1/q:.2f}" for q in ax1.get_xticks()])
    ax2.set_xlabel('Real space resolution ($\mathrm{\AA}$)', fontsize=14)

    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
    if closefig:
        plt.close(fig)

def display_images(images, columns, vmax=None):
    """
    Display images in a grid format.
    
    Parameters:
    - images: A list of images. Each image should be of shape (b, b).
    - columns: Number of columns for displaying images.
    """
    images = convert_to_numpy(images)
    N = len(images)
    rows = N // columns
    rows += N % columns

    position = range(1, N + 1)

    fig = plt.figure(figsize=(columns * 3, rows * 3))
    
    for k, image in zip(position, images):
        ax = fig.add_subplot(rows, columns, k)
        ax.imshow(image, cmap='gray', vmax=vmax)
        ax.set_aspect('equal')
        ax.set_title(f'Image {k}')
        plt.axis('off')
    
    plt.show()
    
def display_images_in_parallel(
        tensors1, tensors2, 
        titles=('Predictions', 'True Values'), ax=None, save_to=None, closefig=True):
    """
    Plots two lists of tensors side by side.
    :param tensors1: List of tensors, numpy arrays, or images of shape Nxbxb (or similar shape).
    :param tensors2: List of tensors, numpy arrays, or images of shape Nxbxb (or similar shape).
    :param titles: Tuple of titles for the two tensor lists.
    """

    # Convert tensors to numpy arrays if they aren't already
    tensors1 = [t.detach().cpu().numpy().squeeze(0) if torch.is_tensor(t) else np.array(t).squeeze(0) for t in tensors1]
    tensors2 = [t.detach().cpu().numpy().squeeze(0) if torch.is_tensor(t) else np.array(t).squeeze(0) for t in tensors2]
    assert len(tensors1) == len(tensors2), "Both tensor lists must have the same length"

    N = len(tensors1)
    if ax is None:
        fig, ax = plt.subplots(2, N, figsize=(3 * N, 6.5))
    if N == 1:
        ax[0].imshow(tensors1[0], cmap='gray', vmax=max(tensors1[0].max() * 1e-3, tensors1[0].min()))
        ax[0].set_title(f"{titles[0]} 0")
        ax[0].axis('off')
        ax[1].imshow(tensors2[0], cmap='gray', vmax=max(tensors2[0].max() * 1e-3, tensors2[0].min()))
        ax[1].set_title(f"{titles[1]} 0")
        ax[1].axis('off')
    else:
        for i in range(N):
            ax[0, i].imshow(tensors1[i], cmap='gray', vmax=max(tensors1[i].max() * 1e-3, tensors1[i].min()))
            ax[0, i].set_title(f"{titles[0]} {i}")
            ax[0, i].axis('off')
            ax[1, i].imshow(tensors2[i], cmap='gray', vmax=max(tensors2[i].max() * 1e-3, tensors2[i].min()))
            ax[1, i].set_title(f"{titles[1]} {i}")
            ax[1, i].axis('off')
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
    if closefig:
        plt.close(fig)

def display_volumes(volumes, ax=None, save_to=None, closefig=True, vmin=None, vmax=None, cmap=None):

    if isinstance(volumes, list):
        volumes = [convert_to_numpy(v) for v in volumes]
    else:
        volumes = [convert_to_numpy(volumes)]

    N = len(volumes)
    if ax is None:
        fig, ax = plt.subplots(N, 3, figsize=(9.5, 3 * N))
    if N == 1:
        dim1, dim2, dim3 = volumes[0].shape
        ax[0].imshow(volumes[0][dim1//2,:,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax[1].imshow(volumes[0][:,dim2//2,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax[2].imshow(volumes[0][:,:,dim3//2].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax[0].set_ylabel('$z$')
        ax[1].set_ylabel('$z$')
        ax[2].set_ylabel('$y$')
        ax[0].set_xlabel('$y$')
        ax[1].set_xlabel('$x$')
        ax[2].set_xlabel('$x$')
    else:
        for i in range(N):
            dim1, dim2, dim3 = volumes[i].shape
            ax[i,0].imshow(volumes[i][dim1//2,:,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[i,1].imshow(volumes[i][:,dim2//2,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[i,2].imshow(volumes[i][:,:,dim3//2].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            
            ax[i,0].set_ylabel('$z$')
            ax[i,1].set_ylabel('$z$')
            ax[i,2].set_ylabel('$y$')
        ax[-1,0].set_xlabel('$y$')
        ax[-1,1].set_xlabel('$x$')
        ax[-1,2].set_xlabel('$x$')
        
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
    if closefig:
        plt.close(fig)

# def display_volumes(volumes, ax=None, save_to=None, closefig=True, vmin=None, vmax=None, cmap=None):

#     if isinstance(volumes, list):
#         if isinstance(volumes[0], np.ndarray):
#             volumes = np.stack(volumes)
#         elif isinstance(volumes[0], torch.Tensor):
#             volumes = torch.stack(volumes).detach().cpu().numpy()
#     if volumes.ndim == 3:
#         dim1, dim2, dim3 = volumes.shape
#         volumes = [volumes.detach().cpu().numpy() if torch.is_tensor(volumes) else np.array(volumes)]
#     elif volumes.ndim == 4:
#         dim1, dim2, dim3 = volumes.shape[1:]
#         volumes = [v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v) for v in volumes]
        
#     N = len(volumes)
#     if ax is None:
#         fig, ax = plt.subplots(N, 3, figsize=(9.5, 3 * N))
#     if N == 1:
#         ax[0].imshow(volumes[0][dim1//2,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
#         ax[1].imshow(volumes[0][:,dim2//2,:], cmap=cmap, vmin=vmin, vmax=vmax)
#         ax[2].imshow(volumes[0][:,:,dim3//2], cmap=cmap, vmin=vmin, vmax=vmax)
#     else:
#         for i in range(N):
#             ax[i,0].imshow(volumes[i][dim1//2,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
#             ax[i,1].imshow(volumes[i][:,dim2//2,:], cmap=cmap, vmin=vmin, vmax=vmax)
#             ax[i,2].imshow(volumes[i][:,:,dim3//2], cmap=cmap, vmin=vmin, vmax=vmax)
#     plt.tight_layout()
#     # plt.show()
#     if save_to is not None:
#         fig.savefig(save_to, bbox_inches='tight')
#     if closefig:
#         plt.close(fig)