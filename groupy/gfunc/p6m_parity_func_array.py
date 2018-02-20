import groupy.garray.p6m_array as p6a
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

        if ni % 2 == 0 or nj % 2 == 0:
            err_msg = "v must have an uneven number of rows and columns"
            raise ValueError(err_msg)

        if ni != nj:
            err_msg = "The number of rows (ni) must match the number of " \
                      "columns (nj)"
            raise ValueError(err_msg)

        self.vmin = -(nj // 2)
        row_correction = (self.vmin + np.bitwise_and(self.vmin, 1)) // 2
        self.umin = -(ni // 2) - row_correction

        i2g = self.i2g(*v.shape[-2:])

        if v.shape[-3] == 12:
            i2g = i2g.reshape(12, *i2g.shape[-2:])
            self.flat_stabilizer = True
        else:
            self.flat_stabilizer = False

        super(P6MFuncArray, self).__init__(v=v, i2g=i2g)

    def i2g(self, rows, cols):
        """
        Construct i2g mapping
        """
        mirs = 2  # number of mirrorings
        rots = 6  # number of rotations

        i2g = np.zeros((mirs, rots, rows, cols, 4), dtype=np.int)

        # Set stabilizer indices
        for m in range(mirs):
            i2g[m, ..., 0] = m

        for r in range(rots):
            i2g[:, r, ..., 1] = r

        hrows, hcols = rows // 2, cols // 2
        rowmesh, colmesh = np.mgrid[-hrows:hrows+1, -hcols:hcols+1]
        uneven_rows = np.bitwise_and((rows-1) // 2, 1)
        row_correction = uneven_rows * np.bitwise_and(rowmesh, 1)
        i2g[..., -2] = colmesh - rowmesh // 2 - row_correction
        i2g[..., -1] = rowmesh

        return p6a.P6MArray(i2g)

    def g2i(self, g):
        gint = g.reparameterize('axial').data.copy()

        u = gint[..., -2] - self.umin
        v = gint[..., -1] - self.vmin

        gint[..., -2] = v
        gint[..., -1] = u + (v - np.bitwise_and(v, 1)) // 2

        if self.flat_stabilizer:
            gint[..., 1] += gint[..., 0] * 6
            gint = gint[..., 1:]

        return gint
