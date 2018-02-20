
import groupy.garray.Z2_array as z2a
from groupy.gfunc.section_array import SectionArray


class Z2SectionArray(SectionArray):

    def __init__(self, v, rep, umin=None, umax=None, vmin=None, vmax=None):

        if umin is None or umax is None or vmin is None or vmax is None:
            if not (umin is None and umax is None and vmin is None and vmax is None):
                raise ValueError('Either all or none of umin, umax, vmin, vmax must equal None')

            # If (u, v) ranges are not given, determine them from the shape of v,
            # assuming the grid is centered.
            nu, nv = v.shape[-2:]

            hnu = nu // 2
            hnv = nv // 2

            umin = -hnu
            umax = hnu - (nu % 2 == 0)
            vmin = -hnv
            vmax = hnv - (nv % 2 == 0)

        self.umin = umin
        self.umax = umax
        self.vmin = vmin
        self.vmax = vmax

        i2x = z2a.meshgrid(
            u=z2a.u_range(self.umin, self.umax + 1),
            v=z2a.v_range(self.vmin, self.vmax + 1)
        )

        super(Z2SectionArray, self).__init__(v=v, i2x=i2x, rep=rep)

    def x2i(self, x):
        # TODO: check validity of indices and wrap / clamp if necessary
        # (or do this in a separate function, so that this function can be more easily tested?)

        xint = x.reparameterize('int').data.copy()
        xint[..., 0] -= self.umin
        xint[..., 1] -= self.vmin
        return xint
