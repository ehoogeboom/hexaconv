
import groupy.garray.p6m_array as p6ma
from groupy.gfunc.gfuncarray import GFuncArray


class P6MFuncArray(GFuncArray):

    def __init__(self, v, umin=None, umax=None, vmin=None, vmax=None):
        # If (u, v) ranges are not given, determine them from the shape of v,
        # assuming the grid is centered.
        nu, nv = v.shape[-2:]

        if (nu % 2 == 0) or (nv % 2 == 0):
            raise ValueError('Grid should be symmetrical, i.e. u and v should be uneven.')

        hnu = nu // 2
        hnv = nv // 2

        umin = -hnu
        umax = hnu + (nu % 2 == 0)
        vmin = -hnv
        vmax = hnv + (nv % 2 == 0)

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2g = p6ma.meshgrid(
            m=p6ma.m_range(),
            r=p6ma.r_range(0, 6),
            u=p6ma.u_range(self.umin, self.umax + 1),
            v=p6ma.v_range(self.vmin, self.vmax + 1)
        )

        if v.shape[-3] == 12:
            i2g = i2g.reshape(12, i2g.shape[-2], i2g.shape[-1])
            self.flat_stabilizer = True
        else:
            self.flat_stabilizer = False

        super(P6MFuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        gaxial = g.reparameterize('axial').data.copy()
        gaxial[..., 2] -= self.umin
        gaxial[..., 3] -= self.vmin

        if self.flat_stabilizer:
            gaxial[..., 1] += gaxial[..., 0] * 6
            gaxial = gaxial[..., 1:]

        return gaxial


def tst():
    import numpy as np

    x = np.random.randn(2, 6, 9, 9)
    # print x[[1, 2, 3, 4]]

    f = P6MFuncArray(x)

    g = p6ma.P6MArray([1, 0, 0, 0])
    li = f.left_translation_indices(g)
    # print li
    lp = f.left_translation_points(g)

    gfp = f(lp)

    print(gfp)
