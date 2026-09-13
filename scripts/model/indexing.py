import jax.numpy as np

def flatten_3d(matrix):
    """
    Flattens a 3D NumPy array into a 1D array in row-major order.
    """
    return np.ravel(matrix, order='C')  # 'C' ensures row-major order


def to_3d(array_1d, shape):
    """
    Converts a 1D NumPy array back to a 3D array with the given shape.
    """
    return np.reshape(array_1d, shape, order='C')  # 'C' = row-major order


def index3D(idx, dims):
    """
    index3D: Compute the 3D indices (x, y, z) from a 1D index.
    """

    if len(dims) != 3:
        raise ValueError("dims must be a sequence of length 3 specifying the size of the 3D matrix.")

    dimX, dimY, dimZ = dims

    z = idx // (dimX * dimY)
    remXY = idx % (dimX * dimY)
    y = remXY // dimX
    x = remXY % dimX

    return x, y, z


def index1D(x, y, z, dims):
    """
    index1D: Compute the 1D index from 3D indices (x, y, z).
    """

    if len(dims) != 3:
        raise ValueError("dims must be a sequence of length 3 specifying the size of the 3D matrix.")
    
    dimX, dimY, dimZ = dims
    
    # Compute the 1D index (row-major order)
    idx = z * (dimX * dimY) + y * dimX + x
    return idx
