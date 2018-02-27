import numpy as np
import tensorflow as tf

from groupy.gconv.gconv_tensorflow.layers import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
from groupy.gfunc.z2func_array import Z2FuncArray
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.p4mfunc_array import P4MFuncArray
import groupy.garray.C4_array as C4a
import groupy.garray.D4_array as D4a


def test_c4_z2_conv_equivariance():
    im = np.random.randn(2, 5, 5, 1)
    x, y = make_graph(P4ConvZ2, 1)
    check_equivariance(im, x, y, Z2FuncArray, P4FuncArray, C4a)


def test_c4_c4_conv_equivariance():
    im = np.random.randn(2, 5, 5, 4)
    x, y = make_graph(P4ConvP4, 4)
    check_equivariance(im, x, y, P4FuncArray, P4FuncArray, C4a)


def test_d4_z2_conv_equivariance():
    im = np.random.randn(2, 5, 5, 1)
    x, y = make_graph(P4MConvZ2, 1)
    check_equivariance(im, x, y, Z2FuncArray, P4MFuncArray, D4a)


def test_d4_d4_conv_equivariance():
    im = np.random.randn(2, 5, 5, 8)
    x, y = make_graph(P4MConvP4M, 8)
    check_equivariance(im, x, y, P4MFuncArray, P4MFuncArray, D4a)


def make_graph(layer, nti):
    x = tf.placeholder(tf.float32, [None, 5, 5, 1 * nti])
    y = layer(filters=1, kernel_size=3, use_bias=False, padding='same')(x)
    return x, y


def check_equivariance(im, input, output, input_array, output_array, point_group):

    # Transform the image
    f = input_array(im.transpose((0, 3, 1, 2)))
    g = point_group.rand()
    gf = g * f
    im1 = gf.v.transpose((0, 2, 3, 1))

    # Compute
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        yx = sess.run(output, feed_dict={input: im})
        yrx = sess.run(output, feed_dict={input: im1})

    # Transform the computed feature maps
    fmap1_garray = output_array(yrx.transpose((0, 3, 1, 2)))
    r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

    print(np.abs(yx - r_fmap1_data).sum())
    assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)
