from __future__ import division
import numpy as np
from groupy.garray.garray import GArray


class P6Array(GArray):

    parameterizations = ['axial']
    _g_shapes = {'axial': (3,)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'p6'

    def __init__(self, data, p='axial'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[P6Array] = self.__class__.left_action_axial
        super(P6Array, self).__init__(data, p)

    def left_action_axial(self, other):
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('axial').data

        self_r = self_axial[..., 0]
        self_u = self_axial[..., 1]
        self_v = self_axial[..., 2]

        other_r = other_axial[..., 0]
        other_u = other_axial[..., 1]
        other_v = other_axial[..., 2]

        self_r = self_r * np.ones(other_r.shape, dtype=np.int)
        self_u = self_u * np.ones(other_u.shape, dtype=np.int)
        self_v = self_v * np.ones(other_v.shape, dtype=np.int)

        other_r = other_r * np.ones(self_r.shape, dtype=np.int)
        other_u = other_u * np.ones(self_u.shape, dtype=np.int)
        other_v = other_v * np.ones(self_v.shape, dtype=np.int)

        xyz = np.ones((3,) + other_r.shape, dtype=np.int)
        xyz[0] = other_v
        xyz[1] = -(other_v + other_u)
        xyz[2] = other_u

        xyz = rotate_xyz(self_r, xyz)

        new_u, new_v = xyz[2] + self_u, xyz[0] + self_v
        new_r = (other_r + self_r) % 6

        new_axial = np.stack([new_r, new_u, new_v], axis=other_r.ndim)
        return other.factory(new_axial, p='axial')

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
    e = P6Array(np.zeros(shape + (3,), dtype=np.int), 'axial')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size+(3,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 6, size)
    data[..., 1] = np.random.randint(minu, maxu, size)
    data[..., 2] = np.random.randint(minv, maxv, size)
    return P6Array(data=data, p='axial')


def r_range(start=0, stop=6, step=1):
    assert stop > 0
    assert stop <= 6
    assert start >= 0
    assert start < 6
    assert start < stop
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 0] = np.arange(start, stop, step)
    return P6Array(m)


def u_range(start=-1, stop=1, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return P6Array(m)


def v_range(start=-1, stop=1, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return P6Array(m)


def meshgrid(r=r_range(), u=u_range(), v=v_range()):
    r = P6Array(r.data[:, None, None, ...], p=r.p)
    u = P6Array(u.data[None, :, None, ...], p=u.p)
    v = P6Array(v.data[None, None, :, ...], p=v.p)
    return u * v * r

C6 = P6Array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]])


def randC6():
    a = np.random.randint(0, 6)
    return P6Array([a, 0, 0])