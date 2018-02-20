
import groupy.garray.Z3_array as z3a
from groupy.gfunc.section_array import SectionArray


class Z3SectionArray(SectionArray):

    def __init__(self, v, rep=None, umin=None, umax=None, vmin=None, vmax=None, wmin=None, wmax=None):

        if None in [umin, umax, vmin, vmax, wmin, wmax]:
            if not (umin is None and umax is None and vmin is None and vmax is None and wmin is None and wmax is None):
                raise ValueError('Either all or none of umin, umax, vmin, vmax, wmin, wmax must equal None')

            # If (u, v, w) ranges are not given, determine them from the shape of v,
            # assuming the grid is centered.
            nu, nv, nw = v.shape[-3:]

            hnu = nu // 2
            hnv = nv // 2
            hnw = nw // 2

            umin = -hnu
            umax = hnu - (nu % 2 == 0)
            vmin = -hnv
            vmax = hnv - (nv % 2 == 0)
            wmin = -hnw
            wmax = hnw - (nw % 2 == 0)

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax
        self.wmin = wmin
        self.wmax = wmax

        i2x = z3a.meshgrid(
            u=z3a.u_range(self.umin, self.umax + 1),
            v=z3a.v_range(self.vmin, self.vmax + 1),
            w=z3a.w_range(self.wmin, self.wmax + 1)
        )

        super(Z3SectionArray, self).__init__(v=v, i2x=i2x, rep=rep)

    def x2i(self, x):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        xint = x.reparameterize('int').data.copy()
        xint[..., 0] -= self.umin
        xint[..., 1] -= self.vmin
        xint[..., 2] -= self.wmin
        return xint