
import copy
import numpy as np
from groupy.garray.garray import GArray
from groupy.representations.matrix_representation import MatrixRepresentation

# TODO: change order of axes, so that the fiber / feature / channel axis comes before the height / width axes,
# instead of at the end. This would be consistent with the common BCHW order used in CNNs


class SectionArray(object):

    def __init__(self, v, i2x, rep=None):
        """
        An array of sections of a homogeneous vector bundle E.
        A section on E is essentially a vector-valued function f on a homogeneous space X (called the base space),
        but the vectors f(x), f(y) live in different but isomorphic spaces (called the fiber F_x over x).

        The base space of the bundle is a homogeneous space X = G/H, where G is a group and H a subgroup.

        The transformation law of a section typically involves a non-trivial action of H on the fibers, as well as an
        action on the domain X discussed above.
        This action is performed using the representation
         rep : H -> GL(R^K)
        where GL(R^K) denotes the set of real invertible K by K matrices.

        i2x is a GArray that can be thought of as a map
          i2x: I -> X
        that takes indices from the set I of valid indices and produces an element of the base space X.
        One can think of i2x as a "parameterization" of the base-space, although we don't insist on continuity or
        smoothness of this map (which wouldn't make much sense since I is discrete).

        The map i2x is required to be injective (distinct indices are mapped to distinct points in X),
        which means it is invertible on its range (U = range(i2x) = i2x(I) subset-of X).
        The inverse is denoted:
         x2i : U -> I
        The method x2i must be implemented in a subclass that knows the particular space X.
        One can think of x2i as a "chart" on X (lacking continuity and smoothness).

        v is an ndarray that stores the values taken by the sections in this SectionArray.
        We can think of v as a map
         v : J x I -> R^K
        where:
        J is an index set of arbitrary shape, such that each j in J identifies a section.
        I is the index set discussed before (representing points in the base space).
        R^K is the K-dimensional fiber.

        So we have the following diagram:
              i2x
          I <-----> U subset-of X
          |   x2i
        v |
          |
          V
          R^K

        So v implicitly defines a function v' on X (which is nonzero only on U):
        v'(x) = v(x2i(x))

        If we have a map T: E - > E (e.g. left multiplication by g in G), that we want to precompose with v',
         w'(g) = v'(T(x)),
        we can get the corresponding map w by composing maps like this:
        I ---> X ---> X ---> I ---> R^K
          i2x     T     x2i     v
        to obtain the transformed function w : I -> R^K.
        This class knows how to produce such a w as an ndarray that directly maps indices to vectors,
        (and such that the indices correspond to elements in X by the same maps i2x and x2i)

        :param i2x: a GArray of sample points. The sample points are elements of the base space X.
        :param v: a numpy.ndarray of values corresponding to the sample points.
        :param rep: MatrixRepresentation of H.
        """

        if not isinstance(i2x, GArray):
            raise TypeError('i2x must be of type GArray, got' + str(type(i2x)) + ' instead.')

        if not isinstance(v, np.ndarray):
            raise TypeError('v must be of type np.ndarray, got ' + str(type(v)) + ' instead.')

        if not isinstance(rep, MatrixRepresentation) and rep is not None:
            raise TypeError('rep must be of type MatrixRepresentation (or None), got ' + str(type(rep)) + ' instead.')

        if i2x.shape != v.shape[-i2x.ndim:]:
            raise ValueError('The trailing axes of v must match the shape of i2x. Got ' +
                             str(i2x.shape) + ' and ' + str(v.shape) + '.')

        # The fiber axis is the one after the loop / array axes and before the base-space axes
        self._fiber_axis = (-i2x.ndim - 1) % v.ndim

        if rep is not None:
            if v.shape[self._fiber_axis] != rep.dim:
                raise ValueError('Fiber axis -i2x.ndim - 1 = ' + str(-i2x.ndim - 1) +
                                 ' should match rep.dim = ' + str(rep.dim))

        self.i2x = i2x
        self.v = v
        self.rep = rep

    def __call__(self, sample_points):
        """
        Evaluate the sections at the given sample points in the base space
        """
        if not isinstance(sample_points, type(self.i2x)):
            raise TypeError('invalid type ' + str(sample_points))

        si = self.x2i(sample_points)
        inds = [Ellipsis] + [si[..., i] for i in range(si.shape[-1])]
        vi = self.v[inds]
        return vi

    def __getitem__(self, item):
        """
        Get an element (a section) from the array of sections
        """
        # TODO bounds / dim checking
        ret = copy.copy(self)
        ret.v = self.v[item]
        return ret

    def __mul__(self, other):
        # Compute self * other
        # if isinstance(other, GArray):
        #     gp = self.right_translation_points(other)
        #     return self(gp)
        # else:
        #     # Python assumes we *return* NotImplemented instead of raising NotImplementedError,
        #     # when we dont know how to left multiply the given type of object by self.
        return NotImplemented

    def __rmul__(self, other):
        # Compute other * self

        # The action on the base space, permuting the fibers:
        gp = self.left_translation_points(other)
        self_gp = self(gp)

        if self.rep is None:
            # If no representation is specified, we assume a trivial representation.
            # This means we don't have to transform the fiber.
            ret = copy.copy(self)
            ret.v = self_gp
            return ret
        elif isinstance(other, self.rep.group.garray_type):
            # If other is an instance of the group H which acts on the fibers, transform the fibers by rep(other):
            other_self_gp = np.tensordot(self_gp, self.rep(other), axes=[self._fiber_axis, 1])
            other_self_gp = np.rollaxis(other_self_gp, axis=-1, start=self._fiber_axis)

            ret = copy.copy(self)  # TODO: is there a faster way to do this (without copying v)?
            ret.v = other_self_gp
            return ret
        elif isinstance(other, GArray):
            # TODO: when other is not an element of H, we need to split other into a part in H and a part in G/H
            # How this is done depends on the group, so will require a base-class method self.h_proj(other)
            return NotImplemented
        else:
            # Python assumes we *return* NotImplemented instead of raising NotImplementedError,
            # when we dont know how to left multiply the given type of object by self.
            return NotImplemented

    def x2i(self, x):
        raise NotImplementedError()

    def h_proj(self, g):
        raise NotImplementedError()

    def left_translation_points(self, g):
        return g.inv() * self.i2x

    def right_translation_points(self, g):
        return self.i2x * g

    def left_translation_indices(self, g):
        ginv_s = self.left_translation_points(g)
        ginv_s_inds = self.x2i(ginv_s)
        return ginv_s_inds

    def right_translation_indices(self, g):
        sg = self.right_translation_points(g)
        sg_inds = self.x2i(sg)
        return sg_inds

    @property
    def ndim(self):
        return self.v.ndim - self.i2x.ndim - 1

    @property
    def shape(self):
        return self.v.shape[:self.ndim]

    @property
    def f_shape(self):
        return self.i2x.shape

    @property
    def f_ndim(self):
        return self.i2x.ndim

    @property
    def fiber_dim(self):
        return self.v.shape[self._fiber_axis]
