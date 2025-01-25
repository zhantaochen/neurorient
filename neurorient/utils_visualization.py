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
                save_to=None, closefig=False, ax=None, fsc_args={}, fontsize_mid=14):
    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    ax1.plot(q, fsc, **fsc_args)
    ax1.set_xticks(np.linspace(0, np.round(q.max(), decimals=2), 5))
    ax1.set_xlabel('Reciprocal space distance $q$ ($\mathrm{\AA}^{-1}$)', fontsize=fontsize_mid)
    ax1.set_ylabel('Fourier Shell Correlation (FSC)', fontsize=fontsize_mid)

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
        ax2.set_xlabel('Real space resolution ($\mathrm{\AA}$)', fontsize=fontsize_mid)

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
        if isinstance(cmap, (list, tuple)):
            _cmap = cmap[k]
        else:
            _cmap = cmap
        _ax.imshow(image, cmap=_cmap, vmax=vmax)
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
        

def display_images_pcolormesh(images, columns, vmax=None, size=3,
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
        # _ax.imshow(image, cmap=cmap, vmax=vmax)
        # _ax.imshow(image, cmap=cmap, vmax=vmax)
        
        _ax.pcolormesh(image, cmap=cmap, vmax=vmax, linewidth=0, antialiased=True)
        _ax.set_aspect('equal')
        if title != 'none':
            _ax.set_title(title[k])
        _ax.axis('off')
        
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
        
    if not closefig:
        if ax is not None:
            plt.show()
    else:
        plt.close(fig)
    
def display_images_in_parallel(
        tensors1, tensors2, 
        titles=('Predictions', 'True Values'), ax=None, save_to=None, closefig=True, cmap='gray'):
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
        ax[0].imshow(tensors1[0], cmap=cmap, vmax=max(tensors1[0].max() * 5e-3, tensors1[0].min()))
        ax[0].set_title(f"{titles[0]} 0")
        ax[0].axis('off')
        ax[1].imshow(tensors2[0], cmap=cmap, vmax=max(tensors2[0].max() * 5e-3, tensors2[0].min()))
        ax[1].set_title(f"{titles[1]} 0")
        ax[1].axis('off')
    else:
        for i in range(N):
            ax[0, i].imshow(tensors1[i], cmap=cmap, vmax=max(tensors1[i].max() * 5e-3, tensors1[i].min()))
            ax[0, i].set_title(f"{titles[0]} {i}")
            ax[0, i].axis('off')
            ax[1, i].imshow(tensors2[i], cmap=cmap, vmax=max(tensors2[i].max() * 5e-3, tensors2[i].min()))
            ax[1, i].set_title(f"{titles[1]} {i}")
            ax[1, i].axis('off')
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
        plt.close()
    if closefig:
        plt.close()

def display_volumes(volumes, ax=None, save_to=None, closefig=True, vmin=None, vmax=None, cmap=None, axes_labels='xyz', titles=None, 
                    ticklabelssoff=True, fontsize_mid=14):

    if isinstance(volumes, (list, tuple)):
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
        ax[0].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
        ax[1].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
        ax[2].set_ylabel(axes_labels[1], fontsize=fontsize_mid)
        ax[0].set_xlabel(axes_labels[1], fontsize=fontsize_mid)
        ax[1].set_xlabel(axes_labels[0], fontsize=fontsize_mid)
        ax[2].set_xlabel(axes_labels[0], fontsize=fontsize_mid)
        if titles is not None:
            ax[1].set_title(titles[0], fontsize=fontsize_mid)
        if ticklabelssoff:
            for j in range(3):
                ax[j].set_xticks([])
                ax[j].set_yticks([])
    else:
        for i in range(N):
            dim1, dim2, dim3 = volumes[i].shape
            ax[i,0].imshow(volumes[i][dim1//2,:,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[i,1].imshow(volumes[i][:,dim2//2,:].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[i,2].imshow(volumes[i][:,:,dim3//2].T, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            
            ax[i,0].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
            ax[i,1].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
            ax[i,2].set_ylabel(axes_labels[1], fontsize=fontsize_mid)
            if titles is not None:
                ax[i,1].set_title(titles[i], fontsize=fontsize_mid)
        ax[-1,0].set_xlabel(axes_labels[1])
        ax[-1,1].set_xlabel(axes_labels[0])
        ax[-1,2].set_xlabel(axes_labels[0])
        
        if ticklabelssoff:
            for i in range(N):
                for j in range(3):
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
        
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
        plt.close()
    if closefig:
        plt.close()


def display_volumes_pcolormesh(volumes, ax=None, save_to=None, closefig=True, vmin=None, vmax=None, cmap=None, axes_labels='xyz', titles=None, 
                    ticklabelssoff=True, fontsize_mid=14):

    if isinstance(volumes, (list, tuple)):
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
        ax[0].pcolormesh(volumes[0][dim1//2,:,:].T, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
        ax[1].pcolormesh(volumes[0][:,dim2//2,:].T, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
        ax[2].pcolormesh(volumes[0][:,:,dim3//2].T, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
        ax[0].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
        ax[1].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
        ax[2].set_ylabel(axes_labels[1], fontsize=fontsize_mid)
        ax[0].set_xlabel(axes_labels[1], fontsize=fontsize_mid)
        ax[1].set_xlabel(axes_labels[0], fontsize=fontsize_mid)
        ax[2].set_xlabel(axes_labels[0], fontsize=fontsize_mid)
        if titles is not None:
            ax[1].set_title(titles[0], fontsize=fontsize_mid)
        if ticklabelssoff:
            for j in range(3):
                ax[j].set_xticks([])
                ax[j].set_yticks([])
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')
    else:
        for i in range(N):
            dim1, dim2, dim3 = volumes[i].shape
            ax[i,0].pcolormesh(volumes[i][dim1//2,:,:].T, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
            ax[i,1].pcolormesh(volumes[i][:,dim2//2,:].T, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
            ax[i,2].pcolormesh(volumes[i][:,:,dim3//2].T, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
            
            ax[i,0].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
            ax[i,1].set_ylabel(axes_labels[2], fontsize=fontsize_mid)
            ax[i,2].set_ylabel(axes_labels[1], fontsize=fontsize_mid)
            if titles is not None:
                ax[i,1].set_title(titles[i], fontsize=fontsize_mid)
                
            ax[i,0].set_aspect('equal')
            ax[i,1].set_aspect('equal')
            ax[i,2].set_aspect('equal')
        ax[-1,0].set_xlabel(axes_labels[1])
        ax[-1,1].set_xlabel(axes_labels[0])
        ax[-1,2].set_xlabel(axes_labels[0])
        
        if ticklabelssoff:
            for i in range(N):
                for j in range(3):
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
        
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
        plt.close()
    if closefig:
        plt.close()

from e3nn import o3
def _show_single_marker(ax, rotation, marker, cmap, use_degrees=False, s=200, 
                        facecolors=True, edgecolors=True, lw=1):
    alpha, beta, gamma = o3.matrix_to_angles(rotation)
    color = cmap(0.5 + gamma.repeat(2) / 2. / np.pi)[-1]
    if facecolors:
        facecolors = color
    else:
        facecolors = 'none'
    
    if edgecolors:
        edgecolors = color
    else:
        edgecolors = 'none'
        
    if not use_degrees:
        ax.scatter(alpha, beta-np.pi/2, s=s, 
                   edgecolors=edgecolors,
                   facecolors=facecolors, 
                   marker=marker, linewidth=lw)
    else:
        ax.scatter(np.rad2deg(alpha), np.rad2deg(beta-np.pi/2), s=s, 
                   edgecolors=edgecolors, 
                   facecolors=facecolors, 
                   marker=marker, linewidth=lw)
        
    return alpha, beta-np.pi/2, gamma / 2. / np.pi, color

def plot_so3_distribution(probs: torch.Tensor,
                          rots: torch.Tensor,
                          gt_rotation=None,
                          fig=None,
                          ax=None,
                          display_threshold_probability=0.000005,
                          show_color_wheel: bool=True,
                          canonical_rotation=torch.eye(3),
                          cmap: plt.cm = plt.cm.hsv,
                          max_marker_size: float = 10.,
                          min_marker_size: float = None,
                          figsize=(8, 4),
                         ):
    '''
    Taken from https://github.com/google-research/google-research/blob/master/implicit_pdf/evaluation.py

    further taken and modified from https://github.com/dmklee/image2sphere?tab=readme-ov-file
    '''
    # cmap = plt.cm.hsv

    # def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
    #     alpha, beta, gamma = o3.matrix_to_angles(rotation)
    #     color = cmap(0.5 + gamma.repeat(2) / 2. / np.pi)[-1]
    #     ax.scatter(alpha, beta-np.pi/2, s=200, edgecolors=color, facecolors='none', marker=marker, linewidth=1)
    #     return alpha, beta-np.pi/2, gamma / 2. / np.pi, color
        # ax.scatter(alpha, beta-np.pi/2, s=150, edgecolors='k', facecolors='none', marker=marker, linewidth=2)
        # ax.scatter(alpha, beta-np.pi/2, s=250, edgecolors='k', facecolors='none', marker=marker, linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(0.01, 0.08, 0.90, 0.95)
        ax = fig.add_subplot(111, projection='mollweide')

    rots = rots @ canonical_rotation
    alpha, beta, gamma = o3.matrix_to_angles(rots)

    # offset alpha and beta so different gammas are visible
    R = 0.02
    alpha += R * np.cos(gamma)
    beta += R * np.sin(gamma)

    which_to_display = (probs > display_threshold_probability)
    scatterpoint_scaling = max_marker_size / probs[which_to_display].max()
    if min_marker_size is None:
        scatterpoint_sizes = scatterpoint_scaling * probs[which_to_display]
    else:
        probs_normalized = (probs[which_to_display] - probs[which_to_display].min()) / (probs[which_to_display].max() - probs[which_to_display].min())
        scatterpoint_sizes = min_marker_size + (max_marker_size - min_marker_size) * probs_normalized

    # Display the distribution
    ax.scatter(alpha[which_to_display],
               beta[which_to_display]-np.pi/2,
               s=scatterpoint_sizes,
               c=cmap(0.5 + gamma[which_to_display] / 2. / np.pi),
               alpha=0.5)
    alpha_gt = []
    beta_gt = []
    gamma_gt = []
    color_gt = []
    if gt_rotation is not None:
        if isinstance(gt_rotation, list):
            gt_rotation = torch.vstack([_gt_rotation.unsqueeze(0) for _gt_rotation in gt_rotation])
        if len(gt_rotation.shape) == 2:
            gt_rotation = gt_rotation.unsqueeze(0)
        gt_rotation = gt_rotation @ canonical_rotation
        for _gt_rotation in gt_rotation:
            # _show_single_marker(ax, _gt_rotation, 'o')
            _alpha, _beta, _gamma, _color = _show_single_marker(
                ax, _gt_rotation.unsqueeze(0), '*', cmap=cmap)
            alpha_gt.append(_alpha)
            beta_gt.append(_beta)
            gamma_gt.append(_gamma)
            color_gt.append(_color)

    ax.grid(visible=True, which='major')
    # ax.set_xticklabels([])
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    # ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=12)
        ax.spines['polar'].set_visible(True)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)

    # plt.show()

    return {
        'ax': ax,
        'alpha': alpha[which_to_display],
        'beta': beta[which_to_display] - np.pi / 2,
        'gamma': gamma[which_to_display] / 2. / np.pi,
        'sizes': scatterpoint_sizes,
        'colors': cmap(0.5 + gamma[which_to_display] / 2. / np.pi),
        'alpha_gt': alpha_gt,
        'beta_gt': beta_gt,
        'gamma_gt': gamma_gt,
        'colors_gt': color_gt
    }

import matplotlib.ticker as ticker

def plot_so3_distribution_inset(so3_plt_out, fig, ax, 
                                offset_x=0.05, offset_y=0.05, alpha=0.5, s_factor=10,
                                title=None):
    # ax.grid()
    disp_coords = ax.transData.transform((so3_plt_out['alpha'].mean().item(), so3_plt_out['beta'].mean().item()))
    fig_coords = fig.transFigure.inverted().transform(disp_coords)
    ax2 = fig.add_axes([fig_coords[0]+offset_x, fig_coords[1]+offset_y, 0.15, 0.15])
    ax2.scatter(
        np.rad2deg(so3_plt_out['alpha']), 
        np.rad2deg(so3_plt_out['beta']), 
        s=s_factor*so3_plt_out['sizes'], 
        facecolor=so3_plt_out['colors'], edgecolor='none', alpha=alpha
    )
    delta_alpha = 0.25 * (max(np.rad2deg(so3_plt_out['alpha'])) - min(np.rad2deg(so3_plt_out['alpha'])))
    delta_beta = 0.25 * (max(np.rad2deg(so3_plt_out['beta'])) - min(np.rad2deg(so3_plt_out['beta'])))
    
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(3))
    
    ax2.set_xlim(min(np.rad2deg(so3_plt_out['alpha']))-delta_alpha, max(np.rad2deg(so3_plt_out['alpha'])) + delta_alpha)
    ax2.set_ylim(min(np.rad2deg(so3_plt_out['beta']))-delta_beta, max(np.rad2deg(so3_plt_out['beta'])) + delta_beta)
    ax2.set_xticklabels([f'{_tick:.0f}°' for _tick in ax2.get_xticks()])
    ax2.set_yticklabels([f'{_tick:.0f}°' for _tick in ax2.get_yticks()])
    if title is not None:
        ax2.set_title(title)
    return ax2

def plot_so3_distribution_inset_3d(so3_plt_out, fig, ax, offset_x=0.05, offset_y=0.05, alpha=0.5, s_factor=10):
    # ax.grid()
    disp_coords = ax.transData.transform((so3_plt_out['alpha'].mean().item(), so3_plt_out['beta'].mean().item()))
    fig_coords = fig.transFigure.inverted().transform(disp_coords)
    ax2 = fig.add_axes([fig_coords[0]+offset_x, fig_coords[1]+offset_y, 0.15, 0.15], projection='3d')
    ax2.scatter(
        np.rad2deg(so3_plt_out['alpha']), np.rad2deg(so3_plt_out['beta']), np.rad2deg(so3_plt_out['gamma']), 
        s=s_factor*so3_plt_out['sizes'], facecolor=so3_plt_out['colors'], edgecolor='none', alpha=alpha
    )
    delta_alpha = 0.25 * (max(np.rad2deg(so3_plt_out['alpha'])) - min(np.rad2deg(so3_plt_out['alpha'])))
    delta_beta = 0.25 * (max(np.rad2deg(so3_plt_out['beta'])) - min(np.rad2deg(so3_plt_out['beta'])))
    delta_gamma = 0.25 * (max(np.rad2deg(so3_plt_out['gamma'])) - min(np.rad2deg(so3_plt_out['gamma'])))
    ax2.set_xlim(min(np.rad2deg(so3_plt_out['alpha']))-delta_alpha, max(np.rad2deg(so3_plt_out['alpha'])) + delta_alpha)
    ax2.set_ylim(min(np.rad2deg(so3_plt_out['beta']))-delta_beta, max(np.rad2deg(so3_plt_out['beta'])) + delta_beta)
    ax2.set_zlim(min(np.rad2deg(so3_plt_out['gamma']))-delta_gamma, max(np.rad2deg(so3_plt_out['gamma'])) + delta_gamma)
    return ax2
    
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