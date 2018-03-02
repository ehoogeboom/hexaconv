import pytest
import numpy as np
import tensorflow as tf
from chainer import cuda, Variable

from groupy.gconv.gconv_chainer.TransformFilter import TransformGFilter
from groupy.gconv.gconv_tensorflow.transform_kernel import transform_kernel_2d_nhwc
from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices,\
    make_d4_z2_indices, make_d4_p4m_indices, make_c6_p6_indices,\
    make_c6_z2_indices, make_d6_p6m_indices, make_d6_z2_indices


def tf_trans_kernel_nhwc(w, inds):
    no, ni, nti, n, _ = w.shape
    nto = inds.shape[0]
    w = w.transpose((2, 3, 4, 1, 0))

    wt = tf.constant(w)
    rwt = transform_kernel_2d_nhwc(wt, inds)

    with tf.Session() as sess:
        rwt = sess.run(rwt)

    rwt = rwt.transpose(3, 2, 0, 1).reshape(no, nto, ni, nti, n, n)
    return rwt


def ch_trans_filter(w, inds):
    w_gpu = cuda.to_gpu(w)
    inds_gpu = cuda.to_gpu(inds)

    wv = Variable(w_gpu)
    rwv = TransformGFilter(inds_gpu)(wv)

    return cuda.to_cpu(rwv.data)


@pytest.mark.parametrize("make_indices,nti", [
    (make_c4_z2_indices, 1),
    (make_d4_z2_indices, 1),
    (make_c6_z2_indices, 1),
    (make_d6_z2_indices, 1),
    (make_c4_p4_indices, 4),
    (make_d4_p4m_indices, 8),
    (make_c6_p6_indices, 6),
    (make_d6_p6m_indices, 12)
])
@pytest.mark.parametrize("transform", [tf_trans_kernel_nhwc])  # , tf_trans_kernel_nchw])
@pytest.mark.parametrize("ksize", [3, 7])
def test_transforms(make_indices, nti, transform, ksize):
    inds = make_indices(ksize=ksize)

    no, ni = np.random.randint(1, 10, size=2)
    w = np.random.randn(no, ni, nti, ksize, ksize)

    rt = transform(w, inds)
    rc = ch_trans_filter(w, inds)

    np.testing.assert_array_equal(rt, rc)
