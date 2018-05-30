import pytest
import numpy as np
import tensorflow as tf

from groupy.gconv.gconv_tensorflow.layers import P4ConvZ2, P4MConvZ2, P4ConvP4, P4MConvP4M,\
    P6ConvZ2Axial, P6MConvZ2Axial, P6ConvP6Axial, P6MConvP6MAxial
from groupy.gfunc import Z2FuncArray, P4FuncArray, P4MFuncArray
from groupy.gfunc.p6_axial_func_array import P6FuncArray
from groupy.gfunc.p6m_axial_func_array import P6MFuncArray
import groupy.garray.C4_array as c4a
import groupy.garray.D4_array as d4a
import groupy.garray.C6_array as c6a
import groupy.garray.D6_array as d6a
from groupy.hexa.hexa_lattice import hexa_manhattan_dist


def hex_padding(im, margin=6):
    assert im.shape[-2] == im.shape[-1], "Only symmetrical images supported"

    radius = (im.shape[-1] - 1) / 2
    assert radius - margin >= 1, "The image will only zeros at this size"

    for j in range(im.shape[-2]):
        for i in range(im.shape[-1]):
            if hexa_manhattan_dist(0, 0, i-radius, j-radius) > radius - margin:
                im[..., j, i] = np.zeros(im.shape[:-2])
    return im


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


def check_equivariance_hex(im, layers, input_array, output_array, point_group, filters):
    im = hex_padding(im.transpose((0, 3, 1, 2))).transpose((0, 2, 3, 1))
    check_equivariance(im, layers, input_array, output_array, point_group, filters)


# We explicitly initialize the bias to test equivariance for p4/p4m groups
@pytest.mark.parametrize('layer,input_array,output_array,point_group,ndim,eq_check,bias_init', [
    (P4ConvZ2, Z2FuncArray, P4FuncArray, c4a, 1, check_equivariance, tf.random_uniform_initializer),
    (P4MConvZ2, Z2FuncArray, P4MFuncArray, d4a, 1, check_equivariance, tf.random_uniform_initializer),
    (P6ConvZ2Axial, Z2FuncArray, P6FuncArray, c6a, 1, check_equivariance_hex, tf.zeros_initializer),
    (P6MConvZ2Axial, Z2FuncArray, P6MFuncArray, d6a, 1, check_equivariance_hex, tf.zeros_initializer),
    (P4ConvP4, P4FuncArray, P4FuncArray, c4a, 4, check_equivariance, tf.random_uniform_initializer),
    (P4MConvP4M, P4MFuncArray, P4MFuncArray, d4a, 8, check_equivariance, tf.random_uniform_initializer),
    (P6ConvP6Axial, P6FuncArray, P6FuncArray, c6a, 6, check_equivariance_hex, tf.zeros_initializer),
    (P6MConvP6MAxial, P6MFuncArray, P6MFuncArray, d6a, 12, check_equivariance_hex, tf.zeros_initializer),
])
@pytest.mark.parametrize('filters', [1, 3])
@pytest.mark.parametrize('padding', ['same', 'valid'])
def test_equivariance(layer, input_array, output_array, point_group, ndim, eq_check, bias_init, filters, padding):
    im = np.random.randn(1, 15, 15, ndim).astype('float32')
    eq_check(
        im=im,
        layers=[layer(filters, 3, padding=padding, bias_initializer=bias_init)],
        input_array=input_array,
        output_array=output_array,
        point_group=point_group,
        filters=filters
    )


# We explicitly initialize the bias to test equivariance for p4/p4m groups
@pytest.mark.parametrize('layer_1, layer_2,output_array,point_group,eq_check,bias_init', [
    (P4ConvZ2, P4ConvP4, P4FuncArray, c4a, check_equivariance, tf.random_uniform_initializer),
    (P4MConvZ2, P4MConvP4M, P4MFuncArray, d4a, check_equivariance, tf.random_uniform_initializer),
    (P6ConvZ2Axial, P6ConvP6Axial, P6FuncArray, c6a, check_equivariance_hex, tf.zeros_initializer),
    (P6MConvZ2Axial, P6MConvP6MAxial, P6MFuncArray, d6a, check_equivariance_hex, tf.zeros_initializer)
])
@pytest.mark.parametrize('filters', [1, 3])
def test_net_equivariance(layer_1, layer_2, output_array, point_group, eq_check, bias_init, filters):
    im = np.random.randn(1, 15, 15, 1).astype('float32')
    eq_check(
        im=im,
        layers=[
            layer_1(4, 3, padding='same', bias_initializer=bias_init),
            layer_2(filters, 3, padding='same', bias_initializer=bias_init)
            ],
        input_array=Z2FuncArray,
        output_array=output_array,
        point_group=point_group,
        filters=filters
    )
