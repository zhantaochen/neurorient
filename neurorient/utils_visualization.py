import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def display_fsc(q, fsc, 
                resolution=None, criteria=0.5, res_pos=None, show_upper_xlabels=True,
                save_to=None, closefig=False, ax=None, fsc_args={}):
    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    ax1.plot(q, fsc, **fsc_args)
    ax1.set_xticks(np.linspace(0, np.round(q.max(), decimals=2), 5))
    ax1.set_xlabel('Reciprocal space distance $q$ ($\mathrm{\AA}^{-1}$)', fontsize=14)
    ax1.set_ylabel('Fourier Shell Correlation (FSC)', fontsize=14)

    if resolution is not None:
        if isinstance(resolution, (float, int)):
            resolution = [resolution]
        if isinstance(criteria, (float, int)):
            criteria = [criteria]
        if res_pos is None:
            res_pos = ['right'] * len(resolution)
        for res, crit, pos in zip(resolution, criteria, res_pos):
            ax1.hlines(crit, -0.1, 1 / res, linestyles='--', colors='gray', linewidth=1)
            ax1.vlines(1 / res, crit, ax1.get_ylim()[1], linestyles='--', colors='gray', linewidth=1)
            if pos == 'right':
                ax1.text(1 / res + 0.0025, 
                        crit + 0.035, 
                        f'{res:.2f} $\mathrm{{\AA}}$', fontsize=11, ha='left')
            elif pos == 'below':
                ax1.text(1 / res + 0.001, 
                        crit - 0.125, 
                        f'{res:.2f} $\mathrm{{\AA}}$', fontsize=11, ha='right')
    ax1.set_xlim([-0.005, q.max()+0.005])
    ax1.set_ylim([min(-0.05, fsc.min()-0.025), 1.05])

    if show_upper_xlabels:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels([r'Infinity',] + [f"{1/q:.2f}" for q in ax1.get_xticks()[1:]])
        ax2.set_xlabel('Real space resolution ($\mathrm{\AA}$)', fontsize=14)

    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
    if closefig:
        plt.close(fig)

def display_images(images, columns, vmax=None, size=3,
                   gs_kwargs = {'wspace':0, 'hspace':0},
                   cmap='gray', title='auto', save_to=None, closefig=False, ax=None):
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

    if title == 'auto':
        title = [f'Image {k}' for k in position]
    if title != 'none':
        assert len(title) == N, "Number of titles must match number of images"
    
    if ax is None:
        fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * size), 
                                gridspec_kw=gs_kwargs)
        # Flatten the axes for easy looping
        ax_flat = axes.ravel() if rows > 1 or columns > 1 else [axes]
    else:
        gs_sub = gridspec.GridSpecFromSubplotSpec(rows, columns, subplot_spec=ax)
        ax_flat = [plt.subplot(cell) for cell in gs_sub]
    
    for k, (_ax, image) in enumerate(zip(ax_flat, images)):
        _ax.imshow(image, cmap=cmap, vmax=vmax)
        _ax.set_aspect('equal')
        if title != 'none':
            _ax.set_title(title[k])
        _ax.axis('off')
        
    # fig = plt.figure(figsize=(columns * size, rows * size), gs_kwargs=gs_kwargs)
    # for k, image in zip(position, images):
    #     ax = fig.add_subplot(rows, columns, k)
    #     ax.imshow(image, cmap=cmap, vmax=vmax)
    #     ax.set_aspect('equal')
    #     if title != 'none':
    #         ax.set_title(title[k-1])
    #     plt.axis('off')
        
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
        
    if not closefig:
        if ax is not None:
            plt.show()
    else:
        plt.close(fig)
    
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
        ax[0].imshow(tensors1[0], cmap='gray', vmax=max(tensors1[0].max() * 5e-3, tensors1[0].min()))
        ax[0].set_title(f"{titles[0]} 0")
        ax[0].axis('off')
        ax[1].imshow(tensors2[0], cmap='gray', vmax=max(tensors2[0].max() * 5e-3, tensors2[0].min()))
        ax[1].set_title(f"{titles[1]} 0")
        ax[1].axis('off')
    else:
        for i in range(N):
            ax[0, i].imshow(tensors1[i], cmap='gray', vmax=max(tensors1[i].max() * 5e-3, tensors1[i].min()))
            ax[0, i].set_title(f"{titles[0]} {i}")
            ax[0, i].axis('off')
            ax[1, i].imshow(tensors2[i], cmap='gray', vmax=max(tensors2[i].max() * 5e-3, tensors2[i].min()))
            ax[1, i].set_title(f"{titles[1]} {i}")
            ax[1, i].axis('off')
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
    if closefig:
        plt.close(fig)

def display_volumes(volumes, ax=None, save_to=None, closefig=True, vmin=None, vmax=None, cmap=None, axes_labels='xyz'):

    if isinstance(volumes, list):
        volumes = [convert_to_numpy(v) for v in volumes]
    else:
        volumes = [convert_to_numpy(volumes)]

    if axes_labels == 'xyz':
        axes_labels = ['$x$', '$y$', '$z$']
    elif axes_labels == 'hkl':
        axes_labels = ['$h$', '$k$', '$l$']
    
    N = len(volumes)
    if ax is None:
        fig, ax = plt.subplots(N, 3, figsize=(9.5, 3 * N))
    if N == 1:
        dim1, dim2, dim3 = volumes[0].shape
        ax[0].imshow(volumes[0][dim1//2,:,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax[1].imshow(volumes[0][:,dim2//2,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax[2].imshow(volumes[0][:,:,dim3//2].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax[0].set_ylabel(axes_labels[2])
        ax[1].set_ylabel(axes_labels[2])
        ax[2].set_ylabel(axes_labels[1])
        ax[0].set_xlabel(axes_labels[1])
        ax[1].set_xlabel(axes_labels[0])
        ax[2].set_xlabel(axes_labels[0])
    else:
        for i in range(N):
            dim1, dim2, dim3 = volumes[i].shape
            ax[i,0].imshow(volumes[i][dim1//2,:,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[i,1].imshow(volumes[i][:,dim2//2,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[i,2].imshow(volumes[i][:,:,dim3//2].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            
            ax[i,0].set_ylabel(axes_labels[2])
            ax[i,1].set_ylabel(axes_labels[2])
            ax[i,2].set_ylabel(axes_labels[1])
        ax[-1,0].set_xlabel(axes_labels[1])
        ax[-1,1].set_xlabel(axes_labels[0])
        ax[-1,2].set_xlabel(axes_labels[0])
        
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