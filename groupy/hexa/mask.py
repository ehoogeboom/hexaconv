import numpy as np


def hexagon_axial(ksize):
    """
    Returns a mask in the shape of a hexagon in an axial coordinate system.
    """
    assert ksize % 2 == 1

    if ksize == 1:
        return np.ones((1, 1)).astype('float32')

    radius = (ksize-1)/2
    r = np.arange(-radius, radius + 1)
    filter_mask = r[:, None] * r[None, :]
    filter_mask = (filter_mask < radius).astype('float32')
    return filter_mask


def square_axial(ny, nx):
    """
    Returns a mask in the shape of a square in an axial coordinate system.
    Function assumes that the first row is not indented.
    """
    assert nx >= ny, 'Image dimensions ill formatted' \
                     'for axial. nx (%d) < ny (%d)' % (nx, ny)

    mask = np.ones((ny, nx))

    x = np.arange(nx)[None, :] * mask
    y = np.arange(ny)[:, None] * mask

    mask *= -y - 2 * x <= -(ny - 1 - (ny % 2 == 0))
    mask *= (-y - 2 * x <= -(ny - 2))[::-1, ::-1]

    mask = mask.astype('float32')

    return mask
