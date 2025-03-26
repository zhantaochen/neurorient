import torch

import matplotlib.pyplot as plt
import numpy as np
# from .external.image2sphere.so3_utils import so3_healpix_grid
from .so3_relative_angle import so3_relative_angle

from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

import posym
from posym import PointGroup

from tqdm import tqdm

threshold_dict = {
    '0': 60 / 180 * np.pi / 2,
    '1': 30 / 180 * np.pi / 2,
    '2': 15 / 180 * np.pi / 2,
    '3': 7.5 / 180 * np.pi / 2,
    '4': 3.75 / 180 * np.pi / 2,
    '5': 1.875 / 180 * np.pi / 2
}

def so3_unique_subset(point_group, rec_level):
    threshold = threshold_dict[str(rec_level)]
    pg = PointGroup(point_group)
    euler_angles_grid = so3_healpix_grid(rec_level=rec_level)
    matrix_grid = euler_angles_to_matrix(euler_angles_grid.T, convention='YXY')
    assert torch.allclose(matrix_to_euler_angles(matrix_grid, convention='YXY') % (2*np.pi), euler_angles_grid.T)
    
    grid_pt_distance = so3_relative_angle(matrix_grid, matrix_grid)[0]
    matrix_grid = matrix_grid[torch.argsort(grid_pt_distance)]

    num_matrix_grid = matrix_grid.shape[0]
    
    # prepare group action matrices

    op_list_full = []
    for op_class in pg._table.get_all_operations().values():
        op_list_full += op_class
        
    op_list = [op for op in op_list_full if not isinstance(op, posym.operations.identity.Identity)]
        
    op_tensor = torch.zeros(len(op_list), 3, 3)
    for i_op, op in enumerate(op_list):
        op_tensor[i_op] = torch.from_numpy(op.matrix_representation)
    op_tensor_output = torch.zeros(len(op_list_full), 3, 3)
    for i_op, op in enumerate(op_list_full):
        op_tensor_output[i_op] = torch.from_numpy(op.matrix_representation)
        
    rotated_matrix_grid = torch.einsum('gij, pjk -> gpik', op_tensor, matrix_grid)
    
    unique_matrix_mask = torch.ones(matrix_grid.shape[0])

    i_grid = 0
    for i_grid in tqdm(range(num_matrix_grid)):
        if not unique_matrix_mask[i_grid] > 0:
            continue
        relative_angle = so3_relative_angle(rotated_matrix_grid[:,i_grid], matrix_grid)

        indices_to_mark_false = torch.where(relative_angle < threshold)[1]
        indices_to_mark_false = indices_to_mark_false[indices_to_mark_false > i_grid]
        unique_matrix_mask[indices_to_mark_false] *= 0
        
    unique_matrix_mask = unique_matrix_mask.bool()
    
    unique_matrix_grid = matrix_grid[unique_matrix_mask]
    unique_euler_angles_grid = matrix_to_euler_angles(unique_matrix_grid, convention='XYX')
    
    print(f"Unique subset volume ratio: {unique_matrix_grid.sum() / num_matrix_grid}")
    return unique_euler_angles_grid.T, op_tensor_output


def so3_point_group_operations(point_group):
    pg = PointGroup(point_group)

    op_list_full = []
    for op_class in pg._table.get_all_operations().values():
        op_list_full += op_class
        
    op_tensor_output = torch.zeros(len(op_list_full), 3, 3)
    for i_op, op in enumerate(op_list_full):
        op_tensor_output[i_op] = torch.from_numpy(op.matrix_representation)
        
    return op_tensor_output

