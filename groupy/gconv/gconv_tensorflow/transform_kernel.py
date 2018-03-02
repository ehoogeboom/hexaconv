import tensorflow as tf


def transform_kernel_2d_nhwc(kernel, indices):
    """
    Transform a set of filters defined on a split plane group G.
    This is the first step of the G-Conv. The user will typically not have to call this function directly.

    The input filter bank w has shape (nti, n, n, ni, no), where:
    nti: the number of transformations in H (the stabilizer of the origin in the input space)
    For example, nti == 1 for images / functions on Z2, since only the identity translation leaves the origin invariant.
    Similarly, nti == 4 for the group p4, because there are 4 transformations in p4 (namely, the four rotations around
    the origin) that leave the origin in p4 (i.e. the identity transformation) fixed.
    n: the filter width and height
    ni: the number of input channels (note: the input feature map is assumed to have ni * nti number of channels)
    no: the number of output channels (note: the G-Conv will actually create no * nto number of channels, see below.

    The index array has shape (nto, nti, n, n, 3)
    Index arrays for various groups can be created with functions in groupy.gconv.make_gconv_indices.
    For example: make_d4_z2_indices(ksize=3)

    The output filter bank transformed_w has shape (n, n, ni * nti, no * nto),
    (so there are nto times as many filters in the output as we had in the input w)
    """

    nti, w, h, ni, no = kernel.get_shape().as_list()
    transformed_kernel = tf.gather_nd(kernel, indices)                         # shape (nto, nti, n, n, ni, no)

    transformed_kernel = tf.transpose(transformed_kernel, [2, 3, 4, 1, 5, 0])  # shape (n, n, ni, nti, no, nto)
    transformed_kernel = tf.reshape(transformed_kernel, [w, h, ni * nti, -1])  # shape (n, n, ni * nti, no * nto)

    return transformed_kernel
