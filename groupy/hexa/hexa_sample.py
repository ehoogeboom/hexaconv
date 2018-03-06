"""
Resample a 2D Cartesian image to and from the HexaGrid
"""

import numpy as np
from scipy.interpolate import interpn

from groupy.grids.hexa_lattice import zigzaghexa2cartesian, hexa2cartesian, centered_meshgrid, cartesian2hexa


def sample_cartesian2zigzaghexa(f_cartesian, fill_value=0.):

    m1, m2 = np.meshgrid(np.arange(0, f_cartesian.shape[-2]), np.arange(0, f_cartesian.shape[-1]))
    x, y = zigzaghexa2cartesian(m1, m2)

    # f_hex = interpolate_linear(f, X / 10., Y / 10.)
    # X, Y = np.meshgrid(np.arange(0, f.shape[-2]), np.arange(0, f.shape[-1]))
    # f_hex = interp2d(X, Y, f, 'cubic')

    # points = np.r_[[X, Y]].reshape()za
    # griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)[source]

    xi = np.c_[y[..., None], x[..., None]].astype(f_cartesian.dtype)
    f_hex = interpn(
            (np.arange(0, f_cartesian.shape[-2]), np.arange(0, f_cartesian.shape[-1])),
            f_cartesian,
            xi,
            method='splinef2d',
            bounds_error=False,
            fill_value=fill_value
    )

    return f_hex


def sample_zigzaghexa2cartesian(f_hex):
    pass


def sample_cartesian2hexa(f_cartesian, fill_value=0.):

    # A square integer meshgrid converted to hexagonal coordinates results in a parallelogram-shaped section
    # from the hexagonal lattice. In our hexagonal grid, the shape looks like /__/ (skewed in the x / n2 direction)
    # In order to make sure no information in the image is lost, we make sure the paralellogram encloses the square.
    # The height of the parallelogram (size along n1 dimension) is therefore equal to the height of the square,
    # while the width of the parallelogram (at any height) is equal to the height times (1 + 1. / sqrt(3)).
    # This factor can easily be seen by making a drawing and noting that the difference y between the length of the
    # horizontal side of the parallelogram and the side x of the inscribed square is
    # y = x / tan(pi / 3) = x / sqrt(3)

    alpha = 1 + 1. / np.sqrt(3)

    # Map the corners of the image (in Cartesian coordinates) to hexagonal coordinates
    n1_00, n2_00 = cartesian2hexa(x=0, y=0)
    n1_10, n2_10 = cartesian2hexa(x=f_cartesian.shape[-1] - 1, y=0)
    n1_01, n2_01 = cartesian2hexa(x=0, y=f_cartesian.shape[-2] - 1)
    n1_11, n2_11 = cartesian2hexa(x=f_cartesian.shape[-1] - 1, y=f_cartesian.shape[-2] - 1)

    min_n1 = np.floor(np.min([n1_00, n1_10, n1_01, n1_11]))
    max_n1 = np.ceil(np.max([n1_00, n1_10, n1_01, n1_11]))
    min_n2 = np.floor(np.min([n2_00, n2_10, n2_01, n2_11]))
    max_n2 = np.ceil(np.max([n2_00, n2_10, n2_01, n2_11]))
    n1 = np.arange(min_n1, max_n1 + 1)
    n2 = np.arange(min_n2, max_n2 + 1)
    n1, n2 = np.meshgrid(n1, n2)

    """min_n1 = np.min([n1_00, n1_10, n1_01, n1_11])
    max_n1 = np.max([n1_00, n1_10, n1_01, n1_11])
    min_n2 = np.min([n2_00, n2_10, n2_01, n2_11])
    max_n2 = np.max([n2_00, n2_10, n2_01, n2_11])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.scatter([min_n1, min_n1, max_n1, max_n1], [min_n2, max_n2, min_n2, max_n2])

    n1_sz = np.ceil(max_n1 - min_n1) + 1
    n1 = np.linspace(start=min_n1, stop=min_n1 + n1_sz - 1, num=n1_sz, endpoint=True)
    n2_sz = np.ceil(max_n2 - min_n2)
    n2 = np.linspace(start=min_n2, stop=min_n2 + n2_sz - 1, num=n2_sz, endpoint=True)
    n1, n2 = np.meshgrid(n1, n2)"""

    # The lower-right corner of the image (highest x and y coordinate), in hexagonal coordinates
    # sz1, sz2 = cartesian2hexa(x=f_cartesian.shape[-1], y=f_cartesian.shape[-2])

    # sz1 = np.ceil(alpha * f_cartesian.shape[-2])
    # sz2 = f_cartesian.shape[-1]

    # # n1, n2 = np.meshgrid(np.arange(0, f_cartesian.shape[-2]), np.arange(0, f_cartesian.shape[-1]))
    # n1, n2 = np.meshgrid(np.arange(sz1), np.arange(sz2))
    # # n1, n2 = centered_meshgrid(sz1, sz2)
    x, y = hexa2cartesian(n1, n2)
    # x -= sz1 - f_cartesian.shape[-2]

    xi = np.c_[y[..., None], x[..., None]].astype(f_cartesian.dtype)
    # f_hex = interpn(
    #         (np.arange(0, f_cartesian.shape[-2]), np.arange(0, f_cartesian.shape[-1])),
    #         f_cartesian,
    #         xi,
    #         method='nearest', # 'splinef2d',
    #         bounds_error=False,
    #         fill_value=fill_value
    # )
    from scipy.ndimage.interpolation import map_coordinates
    f_hex = map_coordinates(input=f_cartesian, coordinates=xi.T, order=5, mode='constant', cval=fill_value).T

    # return f_hex, n1, n2, x, y
    return f_hex


def sample_hexa2zigzaghexa(f_hexa, fill_value=0.):

    pass


def sample_cartesian2axial(data):
    data_hex = np.zeros(
        data.shape[:-2] + sample_cartesian2hexa(data[0, 0]).shape,
        data.dtype)

    for i in range(data.shape[0]):
        for channel in range(data.shape[1]):
            data_hex[i, channel] = sample_cartesian2hexa(data[i, channel])

    return data_hex
