import math

import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable

from groupy import hexa

class HexConv2D(chainer.Link):
    """
    Group convolution base class for split plane groups.

    A plane group (aka wallpaper group) is a group of distance-preserving transformations that includes two independent
    discrete translations.

    A group is called split (or symmorphic) if every element in this group can be written as the composition of an
    element from the "stabilizer of the origin" and a translation. The stabilizer of the origin consists of those
    transformations in the group that leave the origin fixed. For example, the stabilizer in the rotation-translation
    group p4 is the set of rotations around the origin, which is (isomorphic to) the group C4.

    Most plane groups are split, but some include glide-reflection generators; such groups are not split.
    For split groups G, the G-conv can be split into a "filter transform" and "translational convolution" part.

    Different subclasses of this class implement the filter transform for various groups, while this class implements
    the common functionality.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 pad=0,
                 wscale=1,
                 use_cudnn=True,
                 dtype=np.float32):
        """
        :param in_channels:
        :param out_channels:
        :param ksize:
        :param filter_mask:
        :param stride:
        :param pad:
        :param wscale:
        :param nobias:
        :param use_cudnn:
        :param initialW:
        :param initial_bias:
        :param dtype:
        :return:
        """
        super(HexConv2D, self).__init__()

        self.dtype = np.dtype(dtype)
        if self.dtype != np.float32 and use_cudnn:
            raise FloatingPointError('float64 cudnn convolutions are buggy, see chainer issue #519')

        if not isinstance(ksize, int):
            raise TypeError('ksize must be an integer (only square filters are supported).')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride if hasattr(stride, '__getitem__') else (stride, stride)
        self.pad = pad if hasattr(pad, '__getitem__') else (pad, pad)
        self.use_cudnn = use_cudnn

        w_shape = (self.out_channels, self.in_channels, self.ksize, self.ksize)
        self.add_param(name='W', shape=w_shape, dtype=self.dtype)

        self.W.data[:] = self.xp.random.normal(
            0, wscale * math.sqrt(1. / (self.ksize ** 2 * self.in_channels)),
            w_shape
        ).astype(self.dtype)

        self.add_param(
            name='b',
            shape=self.out_channels,
            dtype=self.dtype
        )
        self.b.data[:] = self.xp.repeat(self.dtype.type(0.), self.out_channels)

        filter_mask = hexa.mask.hexagon_axial(ksize)
        filter_mask = filter_mask[None, None, ...].astype(dtype)
        self.add_persistent('filter_mask', filter_mask)

    def __call__(self, x):
        # Apply a mask to the filters (optional)
        if self.filter_mask is not None:
            w, m = F.broadcast(self.W, Variable(self.filter_mask))
            w = w * m
        else:
            w = self.W

        # Perform the 2D convolution
        y = F.convolution_2d(x, w, b=self.b, stride=self.stride, pad=self.pad,
                             use_cudnn=self.use_cudnn)

        # Get a square shaped mask if it does not yet exist.
        if not hasattr(self, 'output_mask'):
            ny, nx = y.data.shape[-2:]
            self.add_persistent(
                'output_mask',
                self.xp.array(hexa.mask.square_axial(ny, nx)[None, None, ...]))

        y, m = F.broadcast(y, Variable(self.output_mask))
        y = y * m

        return y
