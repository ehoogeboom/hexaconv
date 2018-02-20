from __future__ import division
import numpy as np
from groupy.garray.garray import GArray
from groupy.garray.p6_array import rotate_xyz
from groupy.garray.p6m_array import P6MArray
from groupy.garray.Z2_array import Z2Array


class D6Array(GArray):
    """
    D6 array
    """

    parameterizations = ['axial']
    _g_shapes = {'axial': (2,)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'D6'

    def __init__(self, data, p='axial'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[D6Array] = self.__class__.left_action_axial
        self._left_actions[P6MArray] = self.__class__.left_action_axial_P6M
        self._left_actions[Z2Array] = self.__class__.left_action_axial_Z2
        super(D6Array, self).__init__(data, p)


    def axial_elements_(self, axial_repr, shape):
        """
        Function rewritten from p6m to broadcast any number of parameters.
        """
        shaper = np.ones(shape[:-len(self._g_shapes['axial'])], dtype=np.int)
        return axial_repr * np.ones(shape[:-len(self._g_shapes['axial'])] + (1,), dtype=np.int)

    def left_action_axial(self, other):
        
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('axial').data

        self_parameters = \
            self.axial_elements_(self_axial, other_axial.shape)
        self_m = self_parameters[..., 0]
        self_r = self_parameters[..., 1]

        other_parameters = \
            self.axial_elements_(other_axial, self_axial.shape)
        other_m = other_parameters[..., 0]
        other_r = other_parameters[..., 1]

        # Update rotaton to inverse rotation if other is mirrored
        self_r[other_m == 1] = 6 - self_r[other_m == 1]

        new_m = (other_m + self_m) % 2
        new_r = (other_r + self_r) % 6

        new_axial = np.stack([new_m, new_r], axis=other_r.ndim)
        return other.factory(new_axial, p='axial')

    def left_action_axial_P6M(self, other):
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('axial').data

        self_parameters = \
            self.axial_elements_(self_axial, other_axial.shape)
        self_m = self_parameters[..., 0]
        self_r = self_parameters[..., 1]

        other_parameters = \
            self.axial_elements_(other_axial, self_axial.shape)
        other_m = other_parameters[..., 0]
        other_r = other_parameters[..., 1]
        other_u = other_parameters[..., 2]
        other_v = other_parameters[..., 3]

        # Use XYZ to compute rotation and mirror
        xyz = uv_to_xyz(other_u, other_v)

        xyz = rotate_xyz(self_r, xyz)
        xyz = mirror_xyz(self_m, xyz)

        # Update rotaton to inverse rotation if other is mirrored
        self_r[other_m == 1] = 6 - self_r[other_m == 1]

        new_m = (other_m + self_m) % 2
        new_r = (other_r + self_r) % 6
        new_v, new_u = xyz[0], xyz[2]

        new_axial = np.stack([new_m, new_r, new_u, new_v], axis=other_r.ndim)
        return other.factory(new_axial, p='axial')

    def left_action_axial_Z2(self, other):
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('int').data

        self_parameters = \
            self.axial_elements_(self_axial, other_axial.shape)
        self_m = self_parameters[..., 0]
        self_r = self_parameters[..., 1]

        other_parameters = \
            self.axial_elements_(other_axial, self_axial.shape)
        other_u = other_parameters[..., 0]
        other_v = other_parameters[..., 1]

        # Use XYZ to compute rotation and mirror
        xyz = uv_to_xyz(other_u, other_v)

        xyz = rotate_xyz(self_r, xyz)
        xyz = mirror_xyz(self_m, xyz)


        new_m = (self_m) % 2
        new_r = (self_r) % 6
        new_v, new_u = xyz[0], xyz[2]

        new_axial = np.stack([new_u, new_v], axis=other_u.ndim)
        return other.factory(new_axial, p='int')

    def inv(self):
        inv_data = self.reparameterize('axial').data.copy()
        m = inv_data[..., 0]
        inv_data[..., 1] = ((-1)**(1-m) * inv_data[..., 1]) % 6
        return self.factory(inv_data, p='axial').reparameterize(self.p)

def mirror_xyz(mirror, xyz):
    xyz = xyz.copy()

    # Add extra dimensions such that the indexing in loop also words for
    # 1-d arrays
    xyz.shape += (1,)

    mask = np.array(mirror == 1)
    mask.shape += (1,) * (mask.ndim == 0)
    R = [xyz[1, mask], xyz[2, mask]]
    xyz[1, mask] = R[1]
    xyz[2, mask] = R[0]

    xyz.shape = xyz.shape[:-1]
    return xyz


def translate_xyz(tu, tv, xyz):
    xyz[0] += tv
    xyz[2] += tu
    return xyz


def uv_to_xyz(u, v):
    xyz = np.ones((3,) + u.shape, dtype=np.int)
    xyz[0] = v
    xyz[1] = -(v + u)
    xyz[2] = u
    return xyz


def identity(shape=(), p='axial'):
    e = D6Array(np.zeros(shape + (2,), dtype=np.int), 'axial')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size+(2,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 6, size)
    return D6Array(data=data, p='axial')


def m_range(start=0, stop=2):
    assert stop > 0
    assert stop <= 2
    assert start >= 0
    assert start < 2
    assert start < stop
    m = np.zeros((stop - start, 2), dtype=np.int)
    m[:, 0] = np.arange(start, stop)
    return D6Array(m)


def r_range(start=0, stop=6, step=1):
    assert stop > 0
    assert stop <= 6
    assert start >= 0
    assert start < 6
    assert start < stop
    m = np.zeros((stop - start, 2), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return D6Array(m)



D6 = D6Array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
               [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])


