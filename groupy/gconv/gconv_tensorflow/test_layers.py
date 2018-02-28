import pytest
import numpy as np
import tensorflow as tf

from groupy.gconv.gconv_tensorflow.layers import P4ConvZ2, P4MConvZ2, P4ConvP4, P4MConvP4M
from groupy.gfunc import Z2FuncArray, P4FuncArray, P4MFuncArray
import groupy.garray.C4_array as c4a
import groupy.garray.D4_array as d4a


@pytest.mark.parametrize('layer,input_array,output_array,point_group,ndim', [
    (P4ConvZ2, Z2FuncArray, P4FuncArray, c4a, 1),
    (P4MConvZ2, Z2FuncArray, P4MFuncArray, d4a, 1),
    (P4ConvP4, P4FuncArray, P4FuncArray, c4a, 4),
    (P4MConvP4M, P4MFuncArray, P4MFuncArray, d4a, 8)
])
@pytest.mark.parametrize('filters', [1, 3])
@pytest.mark.parametrize('kernel_size', [3, 5])
@pytest.mark.parametrize('padding', ['same', 'valid'])
def test_equivariance(layer, input_array, output_array, point_group, ndim, filters, kernel_size, padding):
    im = np.random.randn(1, 15, 15, ndim).astype('float32')
    check_equivariance(
        im=im,
        layers=[layer(filters, kernel_size, padding=padding, use_bias=False)],
        input_array=input_array,
        output_array=output_array,
        point_group=point_group,
        filters=filters
    )


@pytest.mark.parametrize('layer_1, layer_2,input_array,output_array,point_group', [
    (P4ConvZ2, P4ConvP4, Z2FuncArray, P4FuncArray, c4a),
    (P4MConvZ2, P4MConvP4M, Z2FuncArray, P4MFuncArray, d4a),
])
@pytest.mark.parametrize('filters', [1, 3])
def test_net_equivariance(layer_1, layer_2, input_array, output_array, point_group, filters):
    im = np.random.randn(1, 11, 11, 1).astype('float32')
    check_equivariance(
        im=im,
        layers=[
            layer_1(4, 3, use_bias=False),
            layer_2(filters, 3, use_bias=False)
            ],
        input_array=input_array,
        output_array=output_array,
        point_group=point_group,
        filters=filters
    )


def check_equivariance(im, layers, input_array, output_array, point_group, filters):
    # Transform the image
    f = input_array(im.transpose((0, 3, 1, 2)))  # convert to NCHW
    g = point_group.rand()
    gf = g * f
    im1 = gf.v.transpose((0, 2, 3, 1))  # convert back to NHWC

    x = tf.placeholder(tf.float32, im.shape)
    output = x
    for layer in layers:
        output = layer(output)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        fmap = sess.run(output, feed_dict={x: im})
        fmap1 = sess.run(output, feed_dict={x: im1})

    n, h, w, fmap_len = fmap.shape
    # convert to NCHW and expand feature map
    fmap1_garray = output_array(fmap1.transpose((0, 3, 1, 2)).reshape((n, filters, fmap_len // filters, h, w)))
    # flatten feature map andconvert back to NHWC
    r_fmap1_data = (g.inv() * fmap1_garray).v.reshape((n, fmap_len, h, w)).transpose((0, 2, 3, 1))

    assert np.allclose(fmap, r_fmap1_data, rtol=1e-5, atol=1e-3)
