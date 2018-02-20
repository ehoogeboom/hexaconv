from groupy.gconv.make_gconv_indices import \
    make_d6_z2_indices, make_d6_p6m_indices
import splithexgconv2d


class P6MConvZ2(splithexgconv2d.SplitHexGConv2D):
    input_stabilizer_size = 1
    output_stabilizer_size = 12

    def __init__(self, *args, **kwargs):
        super(P6MConvZ2, self).__init__(*args, **kwargs)
        self.initialize_arrays = True

    def make_transformation_indices(self, ksize):
        return make_d6_z2_indices(ksize=ksize)


class P6MConvP6M(splithexgconv2d.SplitHexGConv2D):

    input_stabilizer_size = 12
    output_stabilizer_size = 12

    def __init__(self, *args, **kwargs):
        super(P6MConvP6M, self).__init__(*args, **kwargs)
        self.initialize_arrays = True

    def make_transformation_indices(self, ksize):
        return make_d6_p6m_indices(ksize=ksize)
