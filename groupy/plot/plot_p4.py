import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from groupy.plot.plot_z2 import plot_z2

def make_f(color=0):
    from scipy import ndimage
    f = ndimage.imread('/home/tsc/Projects/se2conv_experiments/f.jpg')  # [1:, 1:]  # Load & make odd size
    ff = np.ones_like(f) * 255
    ff[15:] = f[:485]  # Shift down a bit
    fmask = (ff < 230).sum(-1).astype('bool')  # Binarize

    # c1 = np.array([51, 60, 95])
    # c2 = np.array([100, 55, 94])
    # c3 = np.array([53, 137, 131])

    # Hue rotations
    c1 = np.array([[131, 81, 81]])  # 0
    c2 = np.array([[131, 118, 81]])  # 45
    c3 = np.array([[106, 131, 81]])  # 90
    c4 = np.array([[81, 131, 94]])  # 135
    c5 = np.array([[81, 131, 131]])  # 180
    c6 = np.array([[81, 94, 131]])  # 225
    c7 = np.array([[106, 81, 131]])  # 270
    c8 = np.array([[131, 81, 118]])  # 315
    c = [c1, c2, c3, c4, c5, c6, c7, c8]

    bg = np.array([1, 1, 1]) * 210

    f[fmask, :] = c[color]
    f[np.logical_not(fmask), :] = bg

    return f


import matplotlib.cm as cm
def make_f2(fg=0.5, cmap=cm.viridis):
    from scipy import ndimage

    f = ndimage.imread('/home/tsc/Projects/se2conv_experiments/f.jpg')  # [1:, 1:]  # Load & make odd size
    ff = np.ones_like(f) * 255
    ff[15:] = f[:485]  # Shift down a bit
    fmask = (ff < 230).sum(-1).astype('bool')  # Binarize

    fgc = cmap(fg)[:-1]
    bgc = np.array([1, 1, 1]) * 230

    f[fmask, :] = np.array(fgc) * 255
    f[np.logical_not(fmask), :] = bgc

    return f[1:, 1:]


def make_f3(color=0):
    from scipy import ndimage
    f = ndimage.imread('/home/tsc/Projects/se2conv_experiments/f.jpg')  # [1:, 1:]  # Load & make odd size
    ff = np.ones_like(f) * 255
    ff[15:] = f[:485]  # Shift down a bit
    fmask = (ff < 230).sum(-1).astype('bool')  # Binarize

    # c1 = np.array([51, 60, 95])
    # c2 = np.array([100, 55, 94])
    # c3 = np.array([53, 137, 131])

    c1 = np.array([[60, 166, 160]])  # 0
    c2 = np.array([[89, 173, 166]])  # 45
    c3 = np.array([[147, 196, 193]])  # 90
    c4 = np.array([[215, 225, 227]])  # 135

    c5 = np.array([[233, 69, 64]])  # 180
    c6 = np.array([[242, 89, 87]])  # 225
    c7 = np.array([[248, 110, 112]])  # 270
    c8 = np.array([[255, 155, 148]])  # 315

    """    c1 = np.array([[61, 150, 212]])  # 0
    c2 = np.array([[170, 65, 141]])  # 45
    c3 = np.array([[209, 109, 35]])  # 90
    c4 = np.array([[255, 228, 0]])  # 135

    c5 = np.array([[171, 206, 241]])  # 180
    c6 = np.array([[214, 115, 183]])  # 225
    c7 = np.array([[230, 155, 95]])  # 270
    c8 = np.array([[255, 237, 112]])  # 315"""

    c = [c1, c2, c3, c4, c5, c6, c7, c8]

    bg = np.array([1, 1, 1]) * 210

    f[fmask, :] = c[color]
    f[np.logical_not(fmask), :] = bg

    return f[1:, 1:]


def paper_plots(i=0, ll=1., ul=0.4, cmap1=cm.Blues, cmap2=cm.Greens):
    # Avoid type-3 fonts for camera ready paper

    import matplotlib
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    # fs1 = [make_f2(c, cmap1) for c in np.linspace(ll, ul, 4)]
    # fs2 = [make_f2(c, cmap2) for c in np.linspace(ll, ul, 4)]
    fs1 = [make_f3(i) for i in range(4)]
    fs2 = [make_f3(i) for i in range(4, 8)]

    p4_fmaps = np.r_[fs1] # [1::2]]
    p4m_fmaps = np.r_[fs1, fs2]

    from groupy.gfunc.p4func_array import P4FuncArray
    from groupy.garray.C4_array import C4Array
    def rotate_p4_func(f, r):
        f = P4FuncArray(f)
        rot = C4Array([r], 'int')
        rot_f = rot * f
        return rot_f.v

    # r_p4_fmaps = rotate_p4_func(p4_fmaps.transpose(3, 0, 1, 2), 1).transpose(1, 2, 3, 0)
    # plot_p4(p4_fmaps, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6))
    # plot_p4(r_p4_fmaps, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6))

    from groupy.gfunc.p4mfunc_array import P4MFuncArray
    from groupy.garray.D4_array import D4Array
    def rotate_flip_p4m_func(f, m, r):
        f = P4MFuncArray(f)
        rf = D4Array([m, r], 'int')
        rf_f = rf * f
        return rf_f.v

    print(p4m_fmaps.shape)
    r_p4m_fmaps = rotate_flip_p4m_func(p4m_fmaps.transpose(3, 0, 1, 2), 0, 1).transpose(1, 2, 3, 0)
    from groupy.plot.plot_p4m import plot_p4m
    # imf.reshape(2, 4, 7, 7), rlabels = 'cayley2', fontsize = 10,
    #          labelpad_factor_1= .2, labelpad_factor_2=.8, labelpad_factor_3=0.5, labelpad_factor_4=1.2, figsize=(2.5, 2.5)
    plot_p4m(p4m_fmaps.reshape(2, 4, 499, 499, 3), rlabels='cayley2', fontsize=10, labelpad_factor_1=0.2, labelpad_factor_2=0.8, labelpad_factor_3=0.5, labelpad_factor_4=1.2,
             figsize=(2.5, 2.5), rcolor='black')
    plt.pause(0.05)
    plot_p4m(r_p4m_fmaps.reshape(2, 4, 499, 499, 3), rlabels='cayley2', fontsize=10, labelpad_factor_1=0.2,
             labelpad_factor_2=0.8, labelpad_factor_3=0.5, labelpad_factor_4=1.2,
             figsize=(2.5, 2.5), rcolor='black')

    # plt.savefig('./p4_fmap_e_mini.eps', format='eps', dpi=600)

    # im, fmaps = testplot(im=f, r=1)
    # plot_p4(fmaps, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6))
    # plt.savefig('./p4_fmap_r_mini.eps', format='eps', dpi=600)


def plot_p4(f, fignum=None, rlabels='cayley', rcolor='red', rlinestyle='-',
            fontsize=20, labelpad_factor_1=1.5, labelpad_factor_2=1.5, figsize=(3, 3)):

    assert rlabels in ['radians', 'cayley', 'indices', 'none']
    assert f.shape[0] == 4
    assert f.ndim == 3 or f.ndim == 4
    ny, nx = f.shape[1:3]

    rlabel_names = {
        'radians': ['$0$', '$\\frac{\pi}{2}$', '$\\pi$', '$\\frac{3 \pi}{2}$'],
        'cayley': ['$e$', '$r$', '$r^2$', '$r^3$'],
        'indices': [0, 1, 2, 3],
        'none': ['', '', '', '']
    }

    fig = plt.figure(fignum, figsize=(2 * f.shape[1], 2 * f.shape[2]))
    fignum = fig.number
    main_ax = fig.gca()

    figtr = fig.transFigure.inverted()  # Display -> Figure

    # circle = plt.Circle((0.5, 0.5), .375, color='gray', fill=False, linestyle='--', linewidth=3.)
    # main_ax.add_artist(circle)
    # plt.axis('off')

    """ax.annotate('simple', xy=(2., -1), xycoords='data',
                xytext=(100, 60), textcoords='offset points',
                size=20,
                # bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                # patchB=el,
                                connectionstyle="arc3,rad=0.3"),
                )"""

    ax_e = fig.add_subplot(3, 3, 2)
    plot_z2(f[0], fignum=fignum)
    ax_e.xaxis.set_label_position('bottom')
    ax_e.set_xlabel(rlabel_names[rlabels][0], fontsize=fontsize, labelpad=labelpad_factor_1 * fontsize)
    ax_e.set_xticks([])
    ax_e.set_yticks([])

    ax_r3 = fig.add_subplot(3, 3, 6)
    plot_z2(f[3], fignum=fignum)
    ax_r3.yaxis.set_label_position('left')
    ax_r3.set_ylabel(rlabel_names[rlabels][3], fontsize=fontsize, rotation='horizontal', va='center', labelpad=labelpad_factor_2 * fontsize)
    ax_r3.set_xticks([])
    ax_r3.set_yticks([])

    ax_r2 = fig.add_subplot(3, 3, 8)
    plot_z2(f[2], fignum=fignum)
    ax_r2.xaxis.set_label_position('top')
    ax_r2.set_xlabel(rlabel_names[rlabels][2], fontsize=fontsize, labelpad=labelpad_factor_1 * fontsize)
    ax_r2.set_xticks([])
    ax_r2.set_yticks([])

    ax_r = fig.add_subplot(3, 3, 4)
    plot_z2(f[1], fignum=fignum)
    ax_r.yaxis.set_label_position('right')
    ax_r.set_ylabel(rlabel_names[rlabels][1], fontsize=fontsize, rotation=0, va='center', labelpad=labelpad_factor_2 * fontsize)
    ax_r.set_xticks([])
    ax_r.set_yticks([])

    # Create pixel coordinate in the subplot coordinate systems for each beginning and enpoint of the arrows
    pt_right = (nx - 0.25, ny // 2)
    pt_top = (nx // 2, -0.75)
    pt_bottom = (nx // 2, ny - 0.25)
    pt_left = (-0.75, ny // 2)
    pt_center = (nx // 2, ny // 2)

    # Transform to figure coordinates
    pt_e_r = figtr.transform(ax_e.transData.transform(pt_left))
    pt_r_e = figtr.transform(ax_r.transData.transform(pt_top))

    pt_r_r2 = figtr.transform(ax_r.transData.transform(pt_bottom))
    pt_r2_r = figtr.transform(ax_r2.transData.transform(pt_left))

    pt_r2_r3 = figtr.transform(ax_r2.transData.transform(pt_right))
    pt_r3_r2 = figtr.transform(ax_r3.transData.transform(pt_bottom))

    pt_r3_e = figtr.transform(ax_r3.transData.transform(pt_top))
    pt_e_r3 = figtr.transform(ax_e.transData.transform(pt_right))

    arrow = FancyArrowPatch(
            pt_e_r,
            pt_r_e,
            transform=fig.transFigure,  # Place arrow in figure coord system
            # connectionstyle="arc3,rad=-0.45",
            connectionstyle='angle3, angleA=10, angleB=-100',
            # mutation_scale=1.0,
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_r_r2,
            pt_r2_r,
            transform=fig.transFigure,  # Place arrow in figure coord system
            # connectionstyle="arc3,rad=-0.45",
            connectionstyle='angle3, angleA=100, angleB=170',
            # mutation_scale=1.0,
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_r2_r3,
            pt_r3_r2,
            transform=fig.transFigure,  # Place arrow in figure coord system
            # connectionstyle="arc3,rad=-0.45",
            connectionstyle='angle3, angleA=190, angleB=260',
            # mutation_scale=1.0,
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    arrow = FancyArrowPatch(
            pt_r3_e,
            pt_e_r3,
            transform=fig.transFigure,  # Place arrow in figure coord system
            # connectionstyle="arc3,rad=-0.45",
            connectionstyle='angle3, angleA=280, angleB=-10',
            # mutation_scale=1.0,
            arrowstyle='->,head_length=3.5,head_width=2.5',
            lw='2.0',
            color=rcolor,
            linestyle=rlinestyle,
    )
    fig.patches.append(arrow)

    main_ax.axis('off')

    # main_ax.annotate("", # transform=fig.transFigure,
    #     xy=pt1_2_ax, xycoords='data',
    #     xytext=pt2_1_ax, textcoords='data',
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
    # )

    # fig.set_size_inches((2 * f.shape[1], 2 * f.shape[2]), forward=True)
    fig.set_size_inches(figsize, forward=True)


def paper_plots2():
    import matplotlib
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    im_e, fmaps_e = testplot(r=0)
    im_r, fmaps_r = testplot(r=1)

    plot_p4(fmaps_e, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6))
    plt.savefig('./p4_fmap_e_mini.eps', format='eps', dpi=600)
    plot_p4(fmaps_r, fontsize=10, labelpad_factor_1=.3, labelpad_factor_2=.6, figsize=(1.6, 1.6))
    plt.savefig('./p4_fmap_r_mini.eps', format='eps', dpi=600)

def testplot(im=None, r=0):

    if im is None:
        # im = np.zeros((39, 39), dtype='float32')
        # im[10:30, 14:16] = 1.
        # im[10:12, 14:30] = 1.
        # im[18:20, 14:24] = 1.
        # im = gaussian_filter(im, sigma=1., mode='constant', cval=0.0)
        im = np.zeros((5, 5), dtype='float32')
        im[0:5, 1] = 1.
        im[0, 1:4] = 1.
        im[2, 1:3] = 1.

    # from gfunc.OLD.transform_Z2_func import rotate_z2_func
    from groupy.gfunc.z2func_array import Z2FuncArray
    from groupy.garray.C4_array import C4Array
    def rotate_z2_func(im, r):
        imf = Z2FuncArray(im)
        rot = C4Array([r], 'int')
        rot_imf = rot * imf
        return rot_imf.v

    im = rotate_z2_func(im, r)

    # im = lena()
    # filter = np.array([1, 2, 1])[:, None] * np.array([-1, 0, 1.])[None, :]

    filter1 = np.array([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).astype(np.float32)
    filter2 = rotate_z2_func(filter1, 1)
    filter3 = rotate_z2_func(filter1, 2)
    filter4 = rotate_z2_func(filter1, 3)

    # imf1 = correlate2d(im, filter1, 'valid')
    # imf2 = correlate2d(im, filter2, 'valid')
    # imf3 = correlate2d(im, filter3, 'valid')
    # imf4 = correlate2d(im, filter4, 'valid')
    # imf1 = convolve2d(im, filter1, 'valid')
    # imf2 = convolve2d(im, filter2, 'valid')
    # imf3 = convolve2d(im, filter3, 'valid')
    # imf4 = convolve2d(im, filter4, 'valid')

    from chainer.functions import Convolution2D
    from chainer import Variable
    im = im.astype(np.float32)
    pad = 2
    imf1 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter1)(Variable(im[None, None])).data[0, 0]
    imf2 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter2)(Variable(im[None, None])).data[0, 0]
    imf3 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter3)(Variable(im[None, None])).data[0, 0]
    imf4 = Convolution2D(in_channels=1, out_channels=1, ksize=3, bias=0., pad=pad, initialW=filter4)(Variable(im[None, None])).data[0, 0]

    return im, np.r_[[imf1, imf2, imf3, imf4]]
