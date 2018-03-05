import numpy as np
from chainer import cuda, Variable


def hex_padding(im, margin=6):
    from groupy.hexa.hexa_lattice import hexa_manhattan_dist
    assert im.shape[-2] == im.shape[-1], "Only symmetrical images supported"

    radius = (im.shape[-1] - 1) / 2
    assert radius - margin >= 1, "The image will only zeros at this size"

    for j in range(im.shape[-2]):
        for i in range(im.shape[-1]):
            if hexa_manhattan_dist(0, 0, i-radius, j-radius) > radius - margin:
                im[..., j, i] = np.zeros(im.shape[:-2])
    return im


def test_p4_net_equivariance():
    from groupy.gfunc import Z2FuncArray, P4FuncArray
    import groupy.garray.C4_array as c4a
    from groupy.gconv.gconv_chainer.p4_conv import P4ConvZ2, P4ConvP4

    im = np.random.randn(1, 1, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[
            P4ConvZ2(in_channels=1, out_channels=2, ksize=3),
            P4ConvP4(in_channels=2, out_channels=3, ksize=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_p6_net_equivariance():
    from groupy.gfunc import Z2FuncArray
    from groupy.gfunc.p6_axial_func_array import P6FuncArray
    import groupy.garray.C6_array as c6a
    from groupy.gconv.gconv_chainer.p6_conv_axial import P6ConvZ2, P6ConvP6

    im = np.random.randn(1, 1, 15, 15).astype('float32')

    check_equivariance_hex(
        im=im,
        layers=[
            P6ConvZ2(in_channels=1, out_channels=2, ksize=3, pad=1),
            P6ConvP6(in_channels=2, out_channels=3, ksize=3, pad=1)
        ],
        input_array=Z2FuncArray,
        output_array=P6FuncArray,
        point_group=c6a,
    )


def test_p4m_net_equivariance():
    from groupy.gfunc import Z2FuncArray, P4MFuncArray
    import groupy.garray.D4_array as d4a
    from groupy.gconv.gconv_chainer.p4m_conv import P4MConvZ2, P4MConvP4M

    im = np.random.randn(1, 1, 11, 11).astype('float32')

    check_equivariance(
        im=im,
        layers=[
            P4MConvZ2(in_channels=1, out_channels=2, ksize=3),
            P4MConvP4M(in_channels=2, out_channels=3, ksize=3)
        ],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def test_p6m_net_equivariance():
    from groupy.gfunc import Z2FuncArray
    from groupy.gfunc.p6m_axial_func_array import P6MFuncArray
    import groupy.garray.D6_array as d6a
    from groupy.gconv.gconv_chainer.p6m_conv_axial import P6MConvZ2, P6MConvP6M

    im = np.random.rand(1, 1, 15, 15).astype('float32')

    check_equivariance_hex(
        im=im,
        layers=[
            P6MConvZ2(in_channels=1, out_channels=1, ksize=3, pad=1),
            P6MConvP6M(in_channels=1, out_channels=1, ksize=3, pad=1)
        ],
        input_array=Z2FuncArray,
        output_array=P6MFuncArray,
        point_group=d6a,
    )


def test_g_z2_conv_equivariance():
    from groupy.gfunc import Z2FuncArray, P4FuncArray, P4MFuncArray
    from groupy.gfunc.p6_axial_func_array import P6FuncArray
    from groupy.gfunc.p6m_axial_func_array import P6MFuncArray

    import groupy.garray.C4_array as c4a
    import groupy.garray.C6_array as c6a
    import groupy.garray.D4_array as d4a
    import groupy.garray.D6_array as d6a
    from groupy.gconv.gconv_chainer.p4_conv import P4ConvZ2
    from groupy.gconv.gconv_chainer.p4m_conv import P4MConvZ2
    from groupy.gconv.gconv_chainer.p6_conv_axial import P6ConvZ2
    from groupy.gconv.gconv_chainer.p6m_conv_axial import P6MConvZ2

    im = np.random.randn(1, 1, 15, 15).astype('float32')
    check_equivariance(
        im=im,
        layers=[P4ConvZ2(1, 2, 3)],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )

    check_equivariance(
        im=im,
        layers=[P4MConvZ2(1, 1, 3)],
        input_array=Z2FuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )

    check_equivariance_hex(
        im=im,
        layers=[P6ConvZ2(1, 2, 3, pad=1)],
        input_array=Z2FuncArray,
        output_array=P6FuncArray,
        point_group=c6a,
    )

    check_equivariance_hex(
        im=im,
        layers=[P6MConvZ2(1, 1, 3, pad=1)],
        input_array=Z2FuncArray,
        output_array=P6MFuncArray,
        point_group=d6a,
    )


def test_p4_p4_conv_equivariance():
    from groupy.gfunc import P4FuncArray
    import groupy.garray.C4_array as c4a
    from groupy.gconv.gconv_chainer.p4_conv import P4ConvP4

    im = np.random.randn(1, 1, 4, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[P4ConvP4(1, 2, 3)],
        input_array=P4FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )


def test_p4m_p4m_conv_equivariance():
    from groupy.gfunc import P4MFuncArray
    import groupy.garray.D4_array as d4a
    from groupy.gconv.gconv_chainer.p4m_conv import P4MConvP4M

    im = np.random.randn(1, 1, 8, 11, 11).astype('float32')
    check_equivariance(
        im=im,
        layers=[P4MConvP4M(1, 2, 3)],
        input_array=P4MFuncArray,
        output_array=P4MFuncArray,
        point_group=d4a,
    )


def test_p6_p6_conv_equivariance():
    from groupy.gfunc.p6_axial_func_array import P6FuncArray
    import groupy.garray.C6_array as c6a
    from groupy.gconv.gconv_chainer.p6_conv_axial import P6ConvP6

    im = np.random.randn(1, 1, 6, 15, 15).astype('float32')
    check_equivariance_hex(
        im=im,
        layers=[P6ConvP6(1, 2, 3, pad=1)],
        input_array=P6FuncArray,
        output_array=P6FuncArray,
        point_group=c6a,
    )


def test_p6m_p6m_conv_equivariance():
    from groupy.gfunc.p6m_axial_func_array import P6MFuncArray
    import groupy.garray.D6_array as d6a
    from groupy.gconv.gconv_chainer.p6m_conv_axial import P6MConvP6M

    im = np.random.randn(1, 1, 12, 15, 15).astype('float32')
    check_equivariance_hex(
        im=im,
        layers=[P6MConvP6M(1, 2, 3, pad=1)],
        input_array=P6MFuncArray,
        output_array=P6MFuncArray,
        point_group=d6a,
    )


def check_equivariance(im, layers, input_array, output_array, point_group):
    np.set_printoptions(threshold=100000)
    # Transform the image
    f = input_array(im)
    g = point_group.rand()
    gf = g * f
    im1 = gf.v

    # Apply layers to both images
    im = Variable(cuda.to_gpu(im))
    im1 = Variable(cuda.to_gpu(im1))

    fmap = im
    fmap1 = im1
    for layer in layers:
        layer.to_gpu()
        fmap = layer(fmap)
        fmap1 = layer(fmap1)

    # Transform the computed feature maps
    fmap1_garray = output_array(cuda.to_cpu(fmap1.data))
    r_fmap1_data = (g.inv() * fmap1_garray).v

    fmap_data = cuda.to_cpu(fmap.data)

    print(fmap_data)
    assert np.allclose(fmap_data, r_fmap1_data, rtol=1e-5, atol=1e-3)


def check_equivariance_hex(im, layers, input_array, output_array, point_group):
    im = hex_padding(im)
    check_equivariance(im, layers, input_array, output_array, point_group)
