import matplotlib
# matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from groupy.grids.hexa_lattice import zigzaghexa2cartesian, centered_meshgrid, hexa_manhattan_dist, zigzaghexa2hexa, hexa2cartesian

# Create image
im = np.zeros((40, 40), dtype='float32')
im[10:30, 14:16] = 1.
im[10:12, 14:30] = 1.
im[18:20, 14:24] = 1.

im2 = np.zeros((10, 10), dtype='float32')
#im2[0] = 1.
#im2[2] = 2.
#im2[4] = 3.
#im2[6] = 4.
#im2[8] = 5.
im2[:, 0] = 1.
im2[:, 2] = 2.
im2[:, 4] = 3.
im2[:, 6] = 4.
im2[:, 8] = 5.

im3 = np.ones((4, 6), dtype='float32')
im3[0] = 2
im3[-1] = 2
im3[:, 0] = 2
im3[:, -1] = 2
# im3[4, 4] = 2


def plot_hexa_im(im, cmap=matplotlib.cm.gray, hex_shape=False, ax=None, zigzag=False, minv=None, maxv=None):
    """
    Plot a hexagonal image. The image is assumed to be represented with the zig-zag parameterization.

    :param im: the image; 2D ndarray
    """
    im[im == im.min()] = np.median(im)

    if zigzag:
        m1, m2 = centered_meshgrid(im.shape[1], im.shape[0])
        x, y = zigzaghexa2cartesian(m1, m2)
    else:
        # a bit strage to use centered_*zigzag*hexa_grid here, but it works
        m1, m2 = centered_meshgrid(im.shape[1], im.shape[0])
        x, y = hexa2cartesian(m1, m2)

    r = (np.minimum(im.shape[0], im.shape[1]) - 1) / 2.

    if ax is None:
        fig, ax = plt.subplots()

    x = x.flatten()
    y = y.flatten()
    m1 = m1.flatten()
    m2 = m2.flatten()
    r = r.flatten()
    im = im.flatten()

    if hex_shape:
        # Circular image; remove hexagons outside radius of the largest circle insribed in the grid
        if zigzag:
            n1, n2 = zigzaghexa2hexa(m1, m2)
        else:
            n1 = m1
            n2 = m2
        dist = hexa_manhattan_dist(n1, n2, 0, 0)
        x = x[dist <= r]
        y = y[dist <= r]
        im = im[dist <= r]

    # Flip the y-axis; in matplotlib's patches code, the y axis points up but we want it to point down
    # as is common in image processing.
    y = np.max(y) - y

    patch_list = [_hex_at(x[i], y[i]) for i in range(x.size)]
    p = PatchCollection(
        patch_list,
        cmap=cmap,
        linewidths=0,
        antialiaseds=10,
    )

    colors = im.reshape(-1, 3) - im.min()
    colors /= colors.max()

    p.set_facecolors(colors)  # set colors
    if minv or maxv:
        p.set_clim([minv, maxv])  # scale color range

    ax.add_collection(p)

    ax.set_xlim(np.min(x) - 0.5, np.max(x) + 0.5)
    ax.set_ylim(np.min(y) - 0.5, np.max(y) + 0.5)
    # plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()


def plot_hexa_ims(ims, cmap=matplotlib.cm.gray, hex_shape=False, zigzag=False):
    assert ims.ndim == 3 or ims.ndim == 4

    if ims.ndim == 3:
        ims = ims[None]

    for i in range(ims.shape[0]):
        for j in range(ims.shape[1]):
            ax = plt.subplot(ims.shape[0], ims.shape[1], i * ims.shape[1] + j + 1)
            plot_hexa_im(ims[i, j], cmap=cmap, hex_shape=hex_shape, ax=ax, zigzag=zigzag)

    plt.tight_layout()


def _hex_at(x, y):

    # The corners of a regular hexagon, with distance between oposite edges equal to 1
    # (the sidelength must be 1/sqrt(3))
    #    2
    # 3 / \ 1
    #   | |
    # 4 \ / 6
    #    5
    halfside = 1. / (2 * np.sqrt(3))  # half the length of a side
    verts = np.array([
       (0.5, halfside),          # 1
       (0.0, 1. / np.sqrt(3)),   # 2
       (-0.5, halfside),         # 3
       (-0.5, -halfside),        # 4
       (0.0, -1. / np.sqrt(3)),  # 5
       (0.5, -halfside),         # 6
       (0., 0.),                 # ignored
    ])

    verts[:, 0] += x
    verts[:, 1] += y

    patch = patches.Polygon(verts[:-1], True)

    return patch

"""
def hexa_plot(f):

    color_map = plt.cm.Spectral_r
    n = 1e5

    # points = np.random.multivariate_normal(mean=(0, 0), cov=np.eye(2), size=int(n))
    # x, y = points.T
    m1, m2 = np.meshgrid(np.arange(0, f.shape[-2]), np.arange(0, f.shape[-1]))
    x, y = zigzaghexa2cartesian(m1, m2)

    min_center_x = np.min(x)
    max_center_x = np.max(x)
    min_center_y = np.min(y)
    max_center_y = np.max(y)

    xbnds = np.array([.0, f.shape[-1]])
    ybnds = np.array([0., f.shape[-2]])
    extent = [xbnds[0], xbnds[1], ybnds[0], ybnds[1]]

    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    print(x.shape, y.shape, f.shape)
    # Set gridsize just to make them visually large
    image = plt.hexbin(
            x.flatten(), y.flatten(),
            C=m2.flatten(),
            #  cmap=color_map,
            gridsize=f.shape[-1],
            extent=extent,
            # mincnt=1,
            # bins='log'
    )

    counts = image.get_array()
    ncnts = np.count_nonzero(np.power(10,counts))
    verts = image.get_offsets()
    for offc in xrange(verts.shape[0]):
        binx, biny = verts[offc][0], verts[offc][1]
        # if counts[offc]:
        plt.plot(binx, biny, 'k.', zorder=100)
    ax.set_xlim(xbnds)
    ax.set_ylim(ybnds)
    plt.grid(True)
    # cb = plt.colorbar(image, spacing='uniform', extend='max')
    plt.show()

    return m1, m2, x, y, image, verts[:, 0], verts[:, 1]
"""
