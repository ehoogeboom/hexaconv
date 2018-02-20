import numpy as np


def test_p4_func():
    from groupy.gfunc.p4func_array import P4FuncArray
    import groupy.garray.C4_array as c4a

    v = np.random.randn(2, 6, 4, 5, 5)
    f = P4FuncArray(v=v)

    g = c4a.rand(size=(1,))
    h = c4a.rand(size=(1,))

    check_associative(g, h, f)
    check_identity(c4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p4m_func():
    from groupy.gfunc.p4mfunc_array import P4MFuncArray
    import groupy.garray.D4_array as d4a

    v = np.random.randn(2, 6, 8, 5, 5)
    f = P4MFuncArray(v=v)

    g = d4a.rand(size=(1,))
    h = d4a.rand(size=(1,))

    check_associative(g, h, f)
    check_identity(d4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6_axial_func():
    from groupy.gfunc.p6_axial_func_array import P6FuncArray
    import groupy.garray.p6_array as p6a

    v = np.random.randn(13, 15, 6, 3, 3)

    v[..., 0, 0] = 0
    v[..., 2, 2] = 0

    f = P6FuncArray(v=v)

    g = p6a.P6Array([np.random.randint(6), 0, 0])
    h = p6a.P6Array([np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6_wide_func():
    from groupy.gfunc.p6_wide_func_array import P6FuncArray
    import groupy.garray.p6_array as p6a

    v = np.random.randn(13, 15, 6, 3, 5)

    v[..., 0, 0] = 0
    v[..., 2, 0] = 0

    f = P6FuncArray(v=v)

    g = p6a.P6Array([np.random.randint(6), 0, 0])
    h = p6a.P6Array([np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6_parity_func():
    from groupy.gfunc.p6_parity_func_array import P6FuncArray
    import groupy.garray.p6_array as p6a

    v = np.random.randn(13, 15, 6, 3, 3)

    v[..., 0, 0] = 0
    v[..., 2, 0] = 0

    f = P6FuncArray(v=v)

    g = p6a.P6Array([np.random.randint(6), 0, 0])
    h = p6a.P6Array([np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6m_axial_func():
    from groupy.gfunc.p6m_axial_func_array import P6MFuncArray
    import groupy.garray.p6m_array as p6ma

    v = np.random.randn(13, 15, 2, 6, 3, 3)

    v[..., 0, 0] = 0
    v[..., 2, 2] = 0

    f = P6MFuncArray(v=v)

    g = p6ma.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])
    h = p6ma.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6ma, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6m_wide_func_flat_stabilizer():
    from groupy.gfunc.p6m_wide_func_array import P6MFuncArray
    import groupy.garray.p6m_array as p6ma

    v = np.random.randn(13, 15, 12, 3, 5)

    v[..., 0, 0] = 0
    v[..., 2, 0] = 0

    f = P6MFuncArray(v=v)

    g = p6ma.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])
    h = p6ma.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6ma, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6m_wide_func():
    from groupy.gfunc.p6m_wide_func_array import P6MFuncArray
    import groupy.garray.p6m_array as p6ma

    v = np.random.randn(13, 15, 2, 6, 3, 5)

    v[..., 0, 0] = 0
    v[..., 2, 0] = 0

    f = P6MFuncArray(v=v)

    g = p6ma.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])
    h = p6ma.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6ma, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6m_parity_func():
    from groupy.gfunc.p6m_parity_func_array import P6MFuncArray
    import groupy.garray.p6m_array as p6a

    v = np.random.randn(13, 15, 2, 6, 3, 3)

    v[..., 0, 0] = 0
    v[..., 2, 0] = 0

    f = P6MFuncArray(v=v)

    g = p6a.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])
    h = p6a.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_p6m_parity_func_flat_stabilizer():
    from groupy.gfunc.p6m_parity_func_array import P6MFuncArray
    import groupy.garray.p6m_array as p6a

    v = np.random.randn(13, 15, 12, 3, 3)

    v[..., 0, 0] = 0
    v[..., 2, 0] = 0

    f = P6MFuncArray(v=v)

    g = p6a.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])
    h = p6a.P6MArray([np.random.randint(2), np.random.randint(6), 0, 0])

    check_associative(g, h, f)
    check_identity(p6a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def test_z2_func():
    from groupy.gfunc.z2func_array import Z2FuncArray
    import groupy.garray.C4_array as c4a
    import groupy.garray.C4_array as d4a

    v = np.random.randn(2, 6, 5, 5)
    f = Z2FuncArray(v=v)

    g = c4a.rand(size=(1,))
    h = c4a.rand(size=(1,))
    check_associative(g, h, f)
    check_identity(c4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)

    g = d4a.rand(size=(1,))
    h = d4a.rand(size=(1,))
    check_associative(g, h, f)
    check_identity(c4a, f)
    check_invertible(g, f)
    check_i2g_g2i_invertible(f)


def check_associative(g, h, f):
    gh = g * h
    hf = h * f
    gh_f = gh * f
    g_hf = g * hf
    assert (gh_f.v == g_hf.v).all()


def check_identity(garray_module, a):
    e = garray_module.identity()
    assert ((e * a).v == a.v).all()


def check_invertible(g, f):
    assert ((g.inv() * (g * f)).v == f.v).all()


def check_i2g_g2i_invertible(f):
    i2g = f.i2g
    i = f.g2i(i2g)
    inds = [i[..., j] for j in range(i.shape[-1])]
    assert (i2g[inds] == i2g).all()





