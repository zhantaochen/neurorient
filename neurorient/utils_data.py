import matplotlib.pyplot as plt
import numpy as np


def uniform_points_on_sphere(n_points, radius=1, seed=None, visualize=False):
    """
    Generate n_points uniformly distributed on a sphere of radius radius.
    """
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n_points, 3)
    points /= np.linalg.norm(points, axis=1)[:, None]
    
    if visualize:
        visualize_points_on_sphere(points)
    
    return points

def visualize_points_on_sphere(points, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
        
        
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    
    ax.plot_wireframe(x, y, z, color='gray', rstride=1, cstride=1, alpha=0.2)
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_box_aspect([1,1,1])  # This ensures the sphere looks like a sphere
    
    plt.show()
