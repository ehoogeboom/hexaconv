"""
Let e1, e2 be an orthogonal basis for R2. The first basis vector e1 points in the horizontal direction, while the
second basis vector e2 points vertically.

A periodic sampling grid is a 2D set of spikes [1]
x = V n,
where V is the Sampling Matrix whose columns are linearly independent sampling vectors,
n = (n1, n2) are the coordinates of the sampling point with respect to the V basis, and
x = (x1, x2) are the coordinates of the sampling point with respect to the (e1, e2) basis.

We will use what Ulichney [1] calls a Hexagonal grid of the first kind, which means that the aspect ratio
  alpha = S / L = 2 / sqrt(3)
where S is the sample period (horizontal distance between samples)
  and L is the line period (distance between horizontal lines of samples)

In terms of sampling matrix, we have
V = [[1, 0.5],
     [0, sqrt(3) / 2]]

Notice that V[:, 0] = e1 and V[:, 1] = R(pi/3) e1 where R(pi/3) is a rotation by pi/3.

Thus, incrementing the first coordinate n1 corresponds to a shift along the horizontal e1 axis of the plane,
while incrementing the second coordinate n2 corresponds to a diagonal shift.
Relative to the standard orthonormal basis for R2, the (n1, n2) cell of the hexagonal lattice is at:
p[n1, n2] = V n = (n1 + 0.5 n2) e1 + 0.5 sqrt(3) n2 e2

The problem with the V-basis is that a square array N of coordinates in this basis corresponds to a parallelogram of
sampling points in the original (e1, e2) basis in which standard Cartesian images are represented.
A more efficient parameterization is what I call the "zig-zag parameterization" where each row is shifted so that
incrementing the n2 coordinate corresponds to a zig-zagging upward motion.

In the zig-zag parameterization of the hexagonal lattice,
incrementing the first coordinate corresponds to a shift along the x axis of the plane,
while the second coordinate corresponds to a zig-zag shift along the second axis.
Relative to the standard orthonormal basis for R2, the (m1, m2) cell of the zig-zag-hexa lattice is at:
p[m1, m2] = (m1 + 0.5 (m2 % 2 == 1)) e1 + 0.5 sqrt(3) m2 e2

References
[1] Ulichney, Digital Halftoning
"""

import numpy as np


def zigzaghexa2cartesian(m1, m2):
    x = m1 + 0.5 * (m2 % 2 == 1)
    y = (0.5 * np.sqrt(3)) * m2
    return x, y


def cartesian2zigzaghexa(x, y):
    m2 = np.round(y / (0.5 * np.sqrt(3)), 0).astype(np.int32)
    m1 = np.round(x - 0.5 * (m2 % 2 == 1), 0).astype(np.int32)
    return m1, m2


def centered_meshgrid(sz1, sz2):
    """
    Create a meshgrid of indices with the center cell corresponding to coordinate (0, 0).
    The index of the center of the array (along one dimension if length sz)
    is sz / 2 for even length arrays and (sz - 1) / 2 for odd length arrays (as returned by central_index).

    :param sz1: size along the m1 dimension
    :param sz2: size along the m2 dimension
    :return:
    """
    hm1 = central_index(sz1)
    hm2 = central_index(sz2)
    m1 = np.arange(-hm1, hm1 + sz1 % 2)
    m2 = np.arange(-hm2, hm2 + sz2 % 2)
    return np.meshgrid(m1, m2)


def central_index(sz):
    return int((sz - sz % 2) / 2.)


def hexa2cartesian(n1, n2):
    x = n1 + 0.5 * n2
    y = (0.5 * np.sqrt(3)) * n2
    return x, y


def cartesian2hexa(x, y):
    n2 = y / (0.5 * np.sqrt(3))
    n1 = x - 0.5 * n2
    return n1, n2


def zigzaghexa2hexa(m1, m2):
    n2 = m2
    n1 = m1 + 0.5 * (m2 % 2 == 1) - 0.5 * m2
    return n1, n2


def hexa2zigzaghexa(n1, n2):
    raise NotImplementedError()


def zigzaghexa_mask(radius):
    n1, n2 = zigzaghexa2hexa(*centered_meshgrid(2 * radius + 1, 2 * radius + 1))
    d = hexa_manhattan_dist(n1, n2, 0, 0)
    return d <= radius


def hexa_mask(radius):
    n1, n2 = centered_meshgrid(2 * radius + 1, 2 * radius + 1)
    d = hexa_manhattan_dist(n1, n2, 0, 0)
    return d <= radius


def hexa_mask_sz(sz1, sz2):
    radius = np.minimum((sz1 - 1) // 2, (sz2 - 1) // 2)
    n1, n2 = centered_meshgrid(sz1, sz2)
    d = hexa_manhattan_dist(n1, n2, 0, 0)
    return d <= radius


def hexa_manhattan_dist(n11, n12, n21, n22):
    """
    Return the Manhattan distance on a hexagonal grid, between points (n11, n12) and (n21, n22).
    The Manhattan distance is the smallest number of discrete steps on the hexagonal lattice that one has to take
     in order to get from one hexagon to another other.
    Both points are assumed to use the standard hexagonal coordinates.

    :param n11: first coordinate of first point
    :param n12: second coordinate of first point
    :param n21: first coordinate of second point
    :param n22: second coordinate of second point
    :return: manhattan distance between the points
    """
    # Adapted from http://stackoverflow.com/questions/5084801/manhattan-distance-between-tiles-in-a-hexagonal-grid
    dx = n11 - n21
    dy = n12 - n22
    d = (np.sign(dx) == np.sign(dy)) * np.abs(dx + dy)
    d += (np.sign(dx) != np.sign(dy)) * np.maximum(np.abs(dx), np.abs(dy))
    return d


def zigzaghexa_manhattan_dist(m11, m12, m21, m22):
    print('untested')  # TODO
    n11, n12 = zigzaghexa2hexa(m11, m12)
    n21, n22 = zigzaghexa2hexa(m21, m22)
    return hexa_manhattan_dist(n11, n12, n21, n22)


def plot_c2o():

    import matplotlib.pyplot as plt

    N = 50
    # x = np.random.rand(N)
    # y = np.random.rand(N)

    minm = 0
    maxm = 6
    m1, m2 = np.meshgrid(np.arange(minm, maxm + 1), np.arange(minm, maxm + 1))

    x, y = zigzaghexa2cartesian(m1, m2)
    print(x.shape)

    plt.scatter(x, y, c=m1)
    plt.figure()
    plt.scatter(x, y, c=m2)
    plt.show()


def test_zigzag():
    for m1 in range(-100, 100):
        for m2 in range(-100, 100):
            mm1, mm2 = cartesian2zigzaghexa(*zigzaghexa2cartesian(m1, m2))
            # print m1, mm1, m2, mm2, 'm1 == mm1', m1 == mm1, m2 == mm2
            assert m1 == mm1
            assert m2 == mm2
