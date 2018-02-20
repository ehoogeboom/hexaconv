
import numpy as np

from groupy.garray.garray import GArray


class Z3Array(GArray):

    parameterizations = ['int']
    _left_actions = {}
    _reparameterizations = {}
    _g_shapes = {'int': (3,)}
    _group_name = 'Z3'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[Z3Array] = self.__class__.z3_composition
        super(Z3Array, self).__init__(data, p)

    def z3_composition(self, other):
        return Z3Array(self.data + other.data)

    def inv(self):
        return Z3Array(-self.data)

    def __repr__(self):
        return 'Z3\n' + self.data.__repr__()

    def reparameterize(self, p):
        assert p == 'int'
        return self


def identity(shape=()):
    e = Z3Array(np.zeros(shape + (3,), dtype=np.int), 'int')
    return e


def rand(minu, maxu, minv, maxv, minw, maxw, size=()):
    data = np.zeros(size + (3,), dtype=np.int64)
    data[..., 0] = np.random.randint(minu, maxu, size)
    data[..., 1] = np.random.randint(minv, maxv, size)
    data[..., 2] = np.random.randint(minw, maxw, size)
    return Z3Array(data=data, p='int')


def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 0] = np.arange(start, stop, step)
    return Z3Array(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return Z3Array(m)


def w_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 3), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return Z3Array(m)


def meshgrid(u=u_range(), v=v_range(), w=w_range()):
    u = Z3Array(u.data[:, None, None, ...], p=u.p)
    v = Z3Array(v.data[None, :, None, ...], p=v.p)
    w = Z3Array(w.data[None, None, :, ...], p=w.p)
    return u * v * w


# def gmeshgrid(*args):
#    out = identity()
#    for i in range(len(args)):
#        slices = [None if j != i else slice(None) for j in range(len(args))] + [Ellipsis]
#        d = args[i].data[slices]
#        print i, slices, d.shape
#        out *= P4MArray(d, p=args[i].p)
#
#    return out
