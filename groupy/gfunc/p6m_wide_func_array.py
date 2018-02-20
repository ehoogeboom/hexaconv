import groupy.garray.p6m_array as p6ma
from groupy.gfunc.gfuncarray import GFuncArray
import numpy as np


class P6MFuncArray(GFuncArray):

    def __init__(self, v):
        """
        We assume the function is defined on a symmetric subset of P6 and
        equals zero for all other elements. Consequently, the elements for
        which the function is defined are 'rings' around (6,6) with arbitrary
        rotation, and thus, we can not accept an arbitrary u and v range.
        Instead the u and v ranges are deduces from v.
        """
        ni, nj = v.shape[-2:]

        if ni % 2 == 0:
            err_msg = "v must have an uneven number of rows"
            raise ValueError(err_msg)

        if ni * 2 - 1 != nj:
            err_msg = "The number of rows (ni) must match the number of " \
                      "columns (nj), i.e., nj = 2*ni - 1"
            raise ValueError(err_msg)

        # TODO: Check whether input v is a 'symmetrical' function.

        hni = ni // 2
        hnj = nj // 4
        hnu, hnv = ij2uv(hni, hnj)

        self.umin = -hnv + hni // 2
        self.vmin = -(hnu+hni)

        i2g = self.i2g(*v.shape[-2:])

        if v.shape[-3] == 12:
            i2g = i2g.reshape(12, *i2g.shape[-2:])
            self.flat_stabilizer = True
        else:
            self.flat_stabilizer = False

        super(P6MFuncArray, self).__init__(v=self.zero_out_v(v), i2g=i2g)

    def i2g(self, imax, jmax):
        """
        Construct i2g mapping
        """
        mirs = 2  # number of mirrorings
        rots = 6  # number of rotations
        unused_col = [0, 1][(imax // 2) % 2 == 1]
        jmax += unused_col

        i2g = np.zeros((mirs, rots, imax, jmax, 4), dtype=np.int)

        # Set stabilizer indices
        for m in range(mirs):
            i2g[m, ..., 0] = m

        for r in range(rots):
            i2g[:, r, ..., 1] = r

        imesh, jmesh = np.mgrid[0:imax, 0:jmax]
        even = (imesh + jmesh) % 2 == 0
        i_, j_ = imesh[even], jmesh[even]

        i2g[..., even, 2] = (j_ - i_)/2 + self.umin
        i2g[..., even, 3] = i_ + self.vmin

        i2g = i2g[..., unused_col:, :]

        return p6ma.P6MArray(i2g)

    def g2i(self, g):
        gint = g.reparameterize('axial').data.copy()

        u = gint[..., -2] - self.umin
        v = gint[..., -1] - self.vmin

        unused_col = (self.v.shape[-2] // 2) % 2 == 1

        gint[..., -2] = v
        gint[..., -1] = u*2 + v - unused_col

        if self.flat_stabilizer:
            gint[..., 1] += gint[..., 0] * 6
            gint = gint[..., 1:]

        return gint

    def __call__(self, sample_points):
        """
        Overwrite super's call such that unused values in v are zeroed out.
        """
        ret = super(P6MFuncArray, self).__call__(sample_points)
        ret.v = self.zero_out_v(ret.v)
        return ret

    def zero_out_v(self, v):
        """
        In the double width representation every other elements in v (and
        g2i for that matter), is not used. This method enforces that the
        values on these indices are always zero.

        Note, v is altered in place.
        """
        unused_col = [0, 1][(v.shape[-2] // 2) % 2 == 1]
        imax, jmax = v.shape[-2:]
        jmax += unused_col
        imesh, jmesh = np.mgrid[0:imax, 0:jmax]
        mask = (imesh + jmesh) % 2 == 1
        mask = mask[..., unused_col:]
        v[..., mask] = 0
        return v


def ij2uv(i, j):
    return (j - i) / 2, i


def uv2ij(u, v):
    return v, u * 2 + v
