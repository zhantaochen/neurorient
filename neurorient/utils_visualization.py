import matplotlib.pyplot as plt
import numpy as np
import torch

def display_images(images, columns, vmax=None):
    """
    Display images in a grid format.
    
    Parameters:
    - images: A list of images. Each image should be of shape (b, b).
    - columns: Number of columns for displaying images.
    """
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
    assert len(tensors1) == len(tensors2), "Both tensor lists must have the same length"

    # Convert tensors to numpy arrays if they aren't already
    tensors1 = [t.detach().cpu().numpy().squeeze() if torch.is_tensor(t) else np.array(t).squeeze() for t in tensors1]
    tensors2 = [t.detach().cpu().numpy().squeeze() if torch.is_tensor(t) else np.array(t).squeeze() for t in tensors2]

    N = len(tensors1)
    if ax is None:
        fig, ax = plt.subplots(2, N, figsize=(3 * N, 6.5))
    for i in range(N):
        ax[0, i].imshow(tensors1[i], cmap='gray', vmax=tensors1[i].max() * 1e-3)
        ax[0, i].set_title(f"{titles[0]} {i}")
        ax[0, i].axis('off')
        ax[1, i].imshow(tensors2[i], cmap='gray', vmax=tensors2[i].max() * 1e-3)
        ax[1, i].set_title(f"{titles[1]} {i}")
        ax[1, i].axis('off')
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
    if closefig:
        plt.close(fig)

def display_volumes(volumes, ax=None, save_to=None, closefig=True):
    if volumes.ndim == 3:
        dim1, dim2, dim3 = volumes.shape
        volumes = [volumes.detach().cpu().numpy() if torch.is_tensor(volumes) else np.array(volumes)]
    elif volumes.ndim == 4:
        dim1, dim2, dim3 = volumes.shape[1:]
        volumes = [v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v) for v in volumes]
        
    N = len(volumes)
    if ax is None:
        fig, ax = plt.subplots(3, N, figsize=(3 * N, 9.5))
    if N == 1:
        ax[0].imshow(volumes[0][dim1//2,:,:], cmap='gray')
        ax[1].imshow(volumes[0][:,dim2//2,:], cmap='gray')
        ax[2].imshow(volumes[0][:,:,dim3//2], cmap='gray')
    else:
        for i in range(N):
            ax[0, i].imshow(volumes[i][dim1//2,:,:], cmap='gray')
            ax[1, i].imshow(volumes[i][:,dim2//2,:], cmap='gray')
            ax[2, i].imshow(volumes[i][:,:,dim3//2], cmap='gray')
    plt.tight_layout()
    # plt.show()
    if save_to is not None:
        fig.savefig(save_to, bbox_inches='tight')
    if closefig:
        plt.close(fig)