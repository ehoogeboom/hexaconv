from __future__ import division
import numpy as np
from groupy.garray.garray import GArray
from groupy.garray.p6_array import rotate_xyz


class P6MArray(GArray):
    """
    P6M wallpaper group element array.

    An element in P6M can be coded using 4 integers (m, r, u, v), where m = {0,
    1} indicated whiter the element is mirrored, r = 1..6 indicates the number
    of rotations, and u and v encode the translation expressed in an axial
    coordinate system, i.e., select an origin and select two of the three axes
    as the u-axis and v-axis. This is called the 'axial' parameterization of
    P6M.

    Using u and v, we can easily compute the coordinate of an element in a cube
    (x, y, z) coordinate system using x = u, y = -(u+v), and z = v. This
    representation is helpful for computing rotations and mirroring.
    Specifically, a CCW rotation of 60 degrees in the XYZ coordinate system is
    computed using

        [x, y, z] ---[ R ]---> [-z, -x, -y] = [x', y', z'].

    Rotations of multiples of 60 degrees or CW rotations can be easily derived
    using this. Similarly, for mirroring the XYZ system can be used, i.e.,

        [x, y, z] ---[ M ]---> [x, z, y] = [x', y', z']

    After these computations we can again obtain the u, v coordinates using
    u = z', v = x'.
    """

    parameterizations = ['axial']
    _g_shapes = {'axial': (4,)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'p6m'

    def __init__(self, data, p='axial'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[P6MArray] = self.__class__.left_action_axial
        super(P6MArray, self).__init__(data, p)

    def axial_elements_(self, axial_repr, shape):
        m = axial_repr[..., 0]
        r = axial_repr[..., 1]
        u = axial_repr[..., 2]
        v = axial_repr[..., 3]

        shaper = np.ones(shape[:-len(self._g_shapes['axial'])], dtype=np.int)
        m = np.array(m * shaper)
        r = np.array(r * shaper)
        u = np.array(u * shaper)
        v = np.array(v * shaper)

        return m, r, u, v

    def left_action_axial(self, other):
        self_axial = self.reparameterize('axial').data
        other_axial = other.reparameterize('axial').data

        self_m, self_r, self_u, self_v = \
            self.axial_elements_(self_axial, other_axial.shape)
        other_m, other_r, other_u, other_v = \
            self.axial_elements_(other_axial, self_axial.shape)

        # Use XYZ to compute rotation and mirror
        xyz = uv_to_xyz(other_u, other_v)

        xyz = rotate_xyz(self_r, xyz)
        xyz = mirror_xyz(self_m, xyz)
        xyz = translate_xyz(self_u, self_v, xyz)

        # Update rotaton to inverse rotation if other is mirrored
        self_r[other_m == 1] = 6 - self_r[other_m == 1]

        new_m = (other_m + self_m) % 2
        new_r = (other_r + self_r) % 6
        new_v, new_u = xyz[0], xyz[2]

        new_axial = np.stack([new_m, new_r, new_u, new_v], axis=other_r.ndim)
        return other.factory(new_axial, p='axial')

    def inv(self):
        inv_data = self.reparameterize('axial').data.copy()
        m = inv_data[..., 0]
        inv_data[..., 1] = ((-1)**(1-m) * inv_data[..., 1]) % 6
        inv_data[..., 2:] = 0
        rot = self.factory(inv_data, p='axial')
        rotated = rot * self
        inv_data[..., 2:] = -rotated.data[..., 2:]
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
    e = P6MArray(np.zeros(shape + (4,), dtype=np.int), 'axial')
    return e.reparameterize(p)


def rand(minu, maxu, minv, maxv, size=()):
    data = np.zeros(size+(4,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 6, size)
    data[..., 2] = np.random.randint(minu, maxu, size)
    data[..., 3] = np.random.randint(minv, maxv, size)
    return P6MArray(data=data, p='axial')


def m_range(start=0, stop=2):
    assert stop > 0
    assert stop <= 2
    assert start >= 0
    assert start < 2
    assert start < stop
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 0] = np.arange(start, stop)
    return P6MArray(m)


def r_range(start=0, stop=6, step=1):
    assert stop > 0
    assert stop <= 6
    assert start >= 0
    assert start < 6
    assert start < stop
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return P6MArray(m)


def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return P6MArray(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 3] = np.arange(start, stop, step)
    return P6MArray(m)


def meshgrid(m=m_range(), r=r_range(), u=u_range(), v=v_range()):
    m = P6MArray(m.data[:, None, None, None, ...], p=m.p)
    r = P6MArray(r.data[None, :, None, None, ...], p=r.p)
    u = P6MArray(u.data[None, None, :, None, ...], p=u.p)
    v = P6MArray(v.data[None, None, None, :, ...], p=v.p)
    return u * v * m * r


D6 = P6MArray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 2, 0, 0], [1, 2, 0, 0],
               [0, 3, 0, 0], [1, 3, 0, 0], [0, 4, 0, 0], [1, 4, 0, 0], [0, 5, 0, 0], [1, 5, 0, 0]])


def randD6():
    a = np.random.randint(0, 2)
    b = np.random.randint(0, 6)

    return P6MArray([a, b, 0, 0])
