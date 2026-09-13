import numpy as np
from scripts.model import indexing

def testcube(domain_dim, offset):
    # Unpack domain dimensions
    matrix_x, matrix_y, matrix_z = domain_dim

    # Unpack offsets
    offset_x, offset_y, offset_z = offset

    # Calculate the size of the inner block of ones
    inner_x = matrix_x - 2 * offset_x
    inner_y = matrix_y - 2 * offset_y
    inner_z = matrix_z - 2 * offset_z

    # Initialize the entire 3D matrix with zeros
    matrix = np.zeros((matrix_x, matrix_y, matrix_z))

    # Set the inner block to ones (True)
    matrix[offset_x : offset_x + inner_x,
           offset_y : offset_y + inner_y,
           offset_z : offset_z + inner_z] = 1

    # flatten to 1D
    C = indexing.flatten_3d(matrix)  

    return C
