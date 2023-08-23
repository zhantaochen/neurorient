import matplotlib.pyplot as plt

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