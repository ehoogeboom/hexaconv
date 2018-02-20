import groupy.garray.p6_array as p6a
from groupy.gfunc.gfuncarray import GFuncArray
import numpy as np


class P6FuncArray(GFuncArray):

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

        super(P6FuncArray, self).__init__(v=v, i2g=i2g)

    def i2g(self, rows, cols):
        """
        Construct i2g mapping
        """
        stabilizer = 6

        i2g = np.zeros((stabilizer, rows, cols, 3), dtype=np.int)

        # Set stabilizer indices
        for s in range(stabilizer):
            i2g[s, ..., 0] = s

        hrows, hcols = rows // 2, cols // 2
        rowmesh, colmesh = np.mgrid[-hrows:hrows+1, -hcols:hcols+1]
        uneven_rows = np.bitwise_and((rows-1) // 2, 1)
        row_correction = uneven_rows * np.bitwise_and(rowmesh, 1)
        i2g[..., 1] = colmesh - rowmesh // 2 - row_correction
        i2g[..., 2] = rowmesh

        return p6a.P6Array(i2g)

    def g2i(self, g):
        gint = g.reparameterize('axial').data.copy()

        u = gint[..., 1] - self.umin
        v = gint[..., 2] - self.vmin

        gint[..., 1] = v
        gint[..., 2] = u + (v - np.bitwise_and(v, 1)) // 2
        return gint
