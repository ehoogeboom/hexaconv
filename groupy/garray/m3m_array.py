import numpy as np
from groupy.garray.matrix_garray import MatrixGArray

from groupy.garray.Z3_array import Z3Array
from groupy.garray.finitegroup import FiniteGroup

# TODO: maybe its better to follow Altmann & Herzig. They use inversion (* -1) instead of reflection in 1 plane. This commutes with all rotations: ir = ri.


class M3MArray(MatrixGArray):
    """
    GArray for the point group
    m3m     (Hermann-Mauguin notation)
    O_h     (Schoenflies notation)
    *432    (orbifold notation)
    [4, 3]  (Coxeter notation)

    This group describes the symmetries of the cube, or equivalently, the symmetries of the octahedron.
    This group is isomorphic to S_4 x C_2, where S_4 acts by permuting the 4 diagonals of the cube, and C_2 acts
    by reflection.
    """
    _a = np.matrix([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])

    _b = np.matrix([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])

    # The matrices of the group O = S_4 of orientation-preserving symmetries of the cube (i.e. rotations)
    # We use this array to map indices (0, ..., 23) to matrices and back.
    _ind2mat = np.array([
        np.eye(3),
        _a,
        _a ** 2,
        _a ** 3,
        _b,
        _b * _a,
        _b * _a ** 2,
        _b * _a ** 3,
        _b ** 2,
        _b ** 2 * _a,
        _b ** 2 * _a ** 2,
        _b ** 2 * _a ** 3,
        _b ** 3,
        _b ** 3 * _a,
        _b ** 3 * _a ** 2,
        _b ** 3 * _a ** 3,
        _a * _b,
        _a * _b * _a,
        _a * _b * _a ** 2,
        _a * _b * _a ** 3,
        _a ** 3 * _b,
        _a ** 3 * _b * _a,
        _a ** 3 * _b * _a ** 2,
        _a ** 3 * _b * _a ** 3
    ]).astype(np.int)

    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (2,), 'mat': (3, 3), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'm3m'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int

        self._left_actions[M3MArray] = self.__class__.left_action_mat
        self._left_actions[Z3Array] = self.__class__.left_action_vec

        super(M3MArray, self).__init__(data, p)

    def int2mat(self, int_data):
        m = int_data[..., 0]
        r = int_data[..., 1]
        out = self._ind2mat[r]
        out[..., 0, :] *= (-1) ** m[..., np.newaxis]
        return out

    def mat2int(self, mat_data):

        # Determine whether the transformation is orientation preserving
        m = (1 - np.linalg.det(mat_data).astype(np.int)) // 2

        # Remove the reflection if needed
        rotations = mat_data.copy()
        rotations[..., 0, :] *= (-1) ** m[..., np.newaxis]

        # Compare the rotation matrices to the rotation matrices in _ind2mat
        # We have:
        # self.data[..., np.newaxis, :, :].shape = (...,  1, 3, 3)
        # _ind2mat.shape =                              (24, 3, 3)
        # equal_elements.shape =                   (..., 24, 3, 3)
        equal_elements = (rotations[..., np.newaxis, :, :] == self._ind2mat)
        equal_mats = equal_elements.prod(axis=(-1, -2))

        # Get the indices of the elements of equal_mats that are 1
        r = np.argmax(equal_mats, axis=-1)

        out = np.zeros(mat_data.shape[:-2] + self._g_shapes['int'], dtype=np.int)
        out[..., 0] = m
        out[..., 1] = r
        return out


class M3MGroup(FiniteGroup, M3MArray):

    def __init__(self):
        # self.garray_type = M3MArray
        # super(D4Group, self).__init__(
        M3MArray.__init__(
            self,
            data=np.c_[np.arange(48) >= 24, np.arange(48) % 24],
            p='int'
        )
        FiniteGroup.__init__(self, M3MArray)

    def factory(self, *args, **kwargs):
        return M3MArray(*args, **kwargs)


m3m = M3MGroup()

# Generators
# TODO


def identity(shape=(), p='int'):
    e = M3MArray(np.zeros(shape + (2,), dtype=np.int), 'int')
    return e.reparameterize(p)


def rand(size=()):
    data = np.zeros(size + (2,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 2, size)
    data[..., 1] = np.random.randint(0, 24, size)
    return M3MArray(data=data, p='int')
