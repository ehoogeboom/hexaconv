from __future__ import division
import numpy as np
from groupy.garray.garray import GArray
from groupy.garray.p6_array import P6Array
from groupy.garray.Z2_array import Z2Array


class C6Array(GArray):

    parameterizations = ['axial']
    _g_shapes = {'axial': (1,)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'c6'

    def __init__(self, data, p='axial'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[Z2Array] = self.__class__.left_action_axial_Z2
        self._left_actions[P6Array] = self.__class__.left_action_axial_P6
        self._left_actions[C6Array] = self.__class__.left_action_axial
        super(C6Array, self).__init__(data, p)

    def left_action_axial(self, other):
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('axial').data

        self_r = self_axial[..., 0]

        other_r = other_axial[..., 0]

        self_r = self_r * np.ones(other_r.shape, dtype=np.int)

        other_r = other_r * np.ones(self_r.shape, dtype=np.int)

        new_r = (other_r + self_r) % 6

        new_axial = np.stack([new_r], axis=other_r.ndim)

        return other.factory(new_axial, p='axial')

    def left_action_axial_P6(self, other):
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('axial').data

        self_r = self_axial[..., 0]

        other_r = other_axial[..., 0]
        other_u = other_axial[..., 1]
        other_v = other_axial[..., 2]

        self_r = self_r * np.ones(other_r.shape, dtype=np.int)

        other_r = other_r * np.ones(self_r.shape, dtype=np.int)
        other_u = other_u * np.ones(self_r.shape, dtype=np.int)
        other_v = other_v * np.ones(self_r.shape, dtype=np.int)

        xyz = np.ones((3,) + other_r.shape, dtype=np.int)
        xyz[0] = other_v
        xyz[1] = -(other_v + other_u)
        xyz[2] = other_u

        xyz = rotate_xyz(self_r, xyz)

        new_u, new_v = xyz[2], xyz[0]
        new_r = (other_r + self_r) % 6

        new_axial = np.stack([new_r, new_u, new_v], axis=other_r.ndim)

        return other.factory(new_axial, p='axial')

    def left_action_axial_Z2(self, other):
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('int').data

        self_r = self_axial[..., 0]

        other_u = other_axial[..., 0]
        other_v = other_axial[..., 1]

        self_r = self_r * np.ones(other_u.shape, dtype=np.int)

        other_u = other_u * np.ones(self_r.shape, dtype=np.int)
        other_v = other_v * np.ones(self_r.shape, dtype=np.int)

        xyz = np.ones((3,) + other_u.shape, dtype=np.int)
        xyz[0] = other_v
        xyz[1] = -(other_v + other_u)
        xyz[2] = other_u

        xyz = rotate_xyz(self_r, xyz)

        new_u, new_v = xyz[2], xyz[0]
        new_r = self_r % 6

        new_axial = np.stack([new_u, new_v], axis=other_u.ndim)

        return other.factory(new_axial, p='int')

    def inv(self):
        inv_data = -self.reparameterize('axial').data.copy()
        inv_data[..., 0] %= 6
        return self.factory(inv_data, p='axial').reparameterize(self.p)

def rotate_xyz(rotation, xyz):
    xyz = xyz.copy()

    rot = rotation % 3
    r_negate = np.array(np.cos(rotation * np.pi))

    # Add extra dimensions such that the indexing in loop also words for
    # 1-d arrays
    xyz.shape += (1,)
    r_negate.shape += (1,)

    for r in [0, 1, 2]:
        mask = np.array(rot == r)
        mask.shape += (1,) * (mask.ndim == 0)
        P = [xyz[0, mask], xyz[1, mask], xyz[2, mask]]
        P = P[-r:] + P[:-r]
        xyz[0, mask] = r_negate[mask] * P[0]
        xyz[1, mask] = r_negate[mask] * P[1]
        xyz[2, mask] = r_negate[mask] * P[2]

    xyz.shape = xyz.shape[:-1]
    return xyz

def identity(shape=(), p='axial'):
    e = C6Array(np.zeros(shape + (1,), dtype=np.int), 'axial')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size+(1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 6, size)
    return C6Array(data=data, p='axial')


C6 = C6Array([[0], [1], [2], [3], [4], [5]])

