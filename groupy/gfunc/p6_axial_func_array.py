
import groupy.garray.p6_array as p6a
from groupy.gfunc.gfuncarray import GFuncArray


class P6FuncArray(GFuncArray):

    def __init__(self, v):

        # If (u, v) ranges are not given, determine them from the shape of v,
        # assuming the grid is centered.
        nu, nv = v.shape[-2:]

        if (nu % 2 == 0) or (nv % 2 == 0):
            raise ValueError('Grid should be symmetrical, i.e. u and v should be uneven.')

        hnu = nu // 2
        hnv = nv // 2

        umin = -hnu
        umax = hnu
        vmin = -hnv
        vmax = hnv

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2g = p6a.meshgrid(
            r=p6a.r_range(0, 6),
            u=p6a.u_range(self.umin, self.umax + 1),
            v=p6a.v_range(self.vmin, self.vmax + 1)
        )

        super(P6FuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gaxial = g.reparameterize('axial').data.copy()
        gaxial[..., 1] -= self.umin
        gaxial[..., 2] -= self.vmin

        return gaxial


def tst():
    import numpy as np

    x = np.random.randn(6, 3, 3)
    # print x[[1, 2, 3, 4]]

    f = P6FuncArray(x)

    g = p6a.P6Array([1, 0, 0])
    li = f.left_translation_indices(g)
    # print li
    lp = f.left_translation_points(g)

    gfp = f(lp)

    print(gfp)
