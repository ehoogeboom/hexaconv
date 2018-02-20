# from groupy.gconv.gconv_chainer.splitgconv2d import SplitGConv2D
from groupy.gconv.make_gconv_indices import \
    make_c6_z2_indices, make_c6_p6_indices

import splithexgconv2d


class Z2ConvZ2(splithexgconv2d.SplitHexGConv2D):
    input_stabilizer_size = 1
    output_stabilizer_size = 1

    def __init__(self, *args, **kwargs):
        super(P6ConvZ2, self).__init__(*args, **kwargs)
        self.initialize_arrays = True

    def make_transformation_indices(self, ksize):
        return make_c6_z2_indices(ksize=ksize)


class P6ConvZ2(splithexgconv2d.SplitHexGConv2D):
    input_stabilizer_size = 1
    output_stabilizer_size = 6

    def __init__(self, *args, **kwargs):
        super(P6ConvZ2, self).__init__(*args, **kwargs)
        self.initialize_arrays = True

    def make_transformation_indices(self, ksize):
        return make_c6_z2_indices(ksize=ksize)


class P6ConvP6(splithexgconv2d.SplitHexGConv2D):
    input_stabilizer_size = 6
    output_stabilizer_size = 6

    def __init__(self, *args, **kwargs):
        super(P6ConvP6, self).__init__(*args, **kwargs)
        self.initialize_arrays = True

    def make_transformation_indices(self, ksize):
        return make_c6_p6_indices(ksize=ksize)
