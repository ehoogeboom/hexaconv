import numpy as np

from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.Z3_array import Z3Array


def _euler_zyz(alpha, beta, gamma):
    za = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha), np.cos(alpha),  0],
                   [0,             0,              1]])
    yb = np.array([[np.cos(beta),  0, -np.sin(beta)],
                   [0,             1,              0],
                   [np.sin(beta),  0,  np.cos(beta)]])
    zg = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma),  0],
                   [0,             0,              1]])
    return za.dot(yb.dot(zg))


class OArray(MatrixGArray):
    """
    GArray for the point group
    432      (Hermann-Mauguin notation)
    O        (Schoenflies notation)
    432      (orbifold notation)
    [4, 3]+  (Coxeter notation)

    This group describes the orientation-preserving symmetries of the cube, or equivalently, those of the octahedron.
    This group is isomorphic to S_4, where S_4 acts by permuting the 4 diagonals of the cube.
    """

    _e = _euler_zyz(0, 0, 0)
    _c2x = _euler_zyz(0, np.pi, np.pi)
    _c2y = _euler_zyz(0, np.pi, 0)
    _c2z = _euler_zyz(0, 0, np.pi)
    _c31p = _euler_zyz(0, np.pi / 2, np.pi / 2)
    _c32p = _euler_zyz(np.pi, np.pi / 2, -np.pi / 2)
    _c33p = _euler_zyz(np.pi, np.pi / 2, np.pi / 2)
    _c34p = _euler_zyz(0, np.pi / 2, -np.pi / 2)
    _c31m = _euler_zyz(np.pi / 2, np.pi / 2, np.pi)
    _c32m = _euler_zyz(-np.pi / 2, np.pi / 2, 0)
    _c33m = _euler_zyz(np.pi / 2, np.pi / 2, 0)
    _c34m = _euler_zyz(-np.pi / 2, np.pi / 2, np.pi)
    _c4xp = _euler_zyz(-np.pi / 2, np.pi / 2, np.pi / 2)
    _c4yp = _euler_zyz(0, np.pi / 2, 0)
    _c4zp = _euler_zyz(0, 0, np.pi / 2)
    _c4xm = _euler_zyz(np.pi / 2, np.pi / 2, -np.pi / 2)
    _c4ym = _euler_zyz(np.pi, np.pi / 2, np.pi)
    _c4zm = _euler_zyz(0, 0, -np.pi / 2)
    _c2ap = _euler_zyz(0, np.pi, np.pi / 2)
    _c2bp = _euler_zyz(0, np.pi, -np.pi / 2)
    _c2cp = _euler_zyz(0, np.pi / 2, np.pi)
    _c2dp = _euler_zyz(np.pi / 2, np.pi / 2, np.pi / 2)
    _c2ep = _euler_zyz(np.pi, np.pi / 2, 0)
    _c2fp = _euler_zyz(-np.pi / 2, np.pi / 2, -np.pi / 2)

    _ind2mat = np.array([
        _e,
        _c2x, _c2y, _c2z,
        _c31p, _c32p, _c33p, _c34p, _c31m, _c32m, _c33m, _c34m,
        _c4xp, _c4yp, _c4zp, _c4xm, _c4ym, _c4zm,
        _c2ap, _c2bp, _c2cp, _c2dp, _c2ep, _c2fp
    ]).astype(np.int)

    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (1,), 'mat': (3, 3), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'O'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[OArray] = self.__class__.left_action_mat
        self._left_actions[Z3Array] = self.__class__.left_action_vec

        super(OArray, self).__init__(data, p)

    def int2mat(self, int_data):
        r = int_data[..., 0]
        out = self._ind2mat[r]
        return out

    def mat2int(self, mat_data):

        # Compare the rotation matrices to the rotation matrices in _ind2mat
        # We have:
        # self.data[..., np.newaxis, :, :].shape = (...,  1, 3, 3)
        # _ind2mat.shape =                              (24, 3, 3)
        # equal_elements.shape =                   (..., 24, 3, 3)
        # equal_mats.shape =                       (..., 24)
        # where the last axis of equal_mats contains zeros except for a single element which is one
        equal_elements = (mat_data[..., np.newaxis, :, :] == self._ind2mat)
        equal_mats = equal_elements.prod(axis=(-1, -2))

        # Get the indices of the elements of equal_mats that are 1
        r = np.argmax(equal_mats, axis=-1)

        out = r.reshape(mat_data.shape[:-2] + self._g_shapes['int'])
        return out


class OGroup(FiniteGroup, OArray):

    def __init__(self):
        OArray.__init__(
            self,
            data=np.arange(24).reshape(24, 1),
            p='int'
        )
        FiniteGroup.__init__(self, OArray)

    def factory(self, *args, **kwargs):
        return OArray(*args, **kwargs)

O = OGroup()


def identity(shape=(), p='int'):
    e = OArray(np.zeros(shape + (1,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size + (1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 24, size)
    return OArray(data=data, p='int')

