
import os

import numpy as np
from groupy.hexa.hexa_sample import sample_cartesian2hexa
from groupy import hexa
import sys


# def global_contrast_normalize(X, scale=55., subtract_mean=True, use_std=False,
#                               sqrt_bias=0., min_divisor=1e-8):
#     """
#     Global contrast normalizes by (optionally) subtracting the mean
#     across features and then normalizes by either the vector norm
#     or the standard deviation (across features, for each example).
#     Parameters
#     ----------
#     X : ndarray, 2-dimensional
#         Design matrix with examples indexed on the first axis and \
#         features indexed on the second.
#     scale : float, optional
#         Multiply features by this const.
#     subtract_mean : bool, optional
#         Remove the mean across features/pixels before normalizing. \
#         Defaults to `True`.
#     use_std : bool, optional
#         Normalize by the per-example standard deviation across features \
#         instead of the vector norm. Defaults to `False`.
#     sqrt_bias : float, optional
#         Fudge factor added inside the square root. Defaults to 0.
#     min_divisor : float, optional
#         If the divisor for an example is less than this value, \
#         do not apply it. Defaults to `1e-8`.
#     Returns
#     -------
#     Xp : ndarray, 2-dimensional
#         The contrast-normalized features.
#     Notes
#     -----
#     `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
#     parameters) corresponds to the preprocessing used in [1].
#     References
#     ----------
#     .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
#        Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
#        http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
#     """
#     assert X.ndim == 2, "X.ndim must be 2"
#     scale = float(scale)
#     assert scale >= min_divisor

#     # Note: this is per-example mean across pixels, not the
#     # per-pixel mean across examples. So it is perfectly fine
#     # to subtract this without worrying about whether the current
#     # object is the train, valid, or test set.
#     mean = X.mean(axis=1)
#     if subtract_mean:
#         X = X - mean[:, np.newaxis]  # Makes a copy.
#     else:
#         X = X.copy()

#     if use_std:
#         # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
#         # Coates' code does.
#         ddof = 1

#         # If we don't do this, X.var will return nan.
#         if X.shape[1] == 1:
#             ddof = 0

#         normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
#     else:
#         normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

#     # Don't normalize by anything too small.
#     normalizers[normalizers < min_divisor] = 1.

#     X /= normalizers[:, np.newaxis]  # Does not make a copy.
#     return X


def normalize(data, eps=1e-8):
    data -= data.mean(axis=(1, 2), keepdims=True)
    std = np.sqrt(data.var(axis=(1, 2), ddof=1, keepdims=True))
    std[std < eps] = 1.
    data /= std
    return data


def get_preprocess_folder(datadir, hex_sampling):
    if hex_sampling == '':
        return os.path.join(datadir, 'preprocessed')
    elif hex_sampling == 'axial':
        return os.path.join(datadir, 'preprocessed_hex_axial')
    else:
        raise ValueError('Unknown option "{}"'.format(hex_sampling))


def save_images(datadir, images, hex_sampling):
    def upsample(image, factor=2):
        shape = image.shape
        shape = (shape[0] * factor, shape[1] * factor, shape[2])
        result = np.zeros(shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = \
                    image[i, j]
        return result

    import matplotlib
    matplotlib.use('Agg')
    from groupy.hexa.hexa_plot_mpl import plot_hexa_im

    if hex_sampling:
        import matplotlib.pyplot as plt
        for i, image in enumerate(images[:5]):
            plot_hexa_im(image.transpose(1, 2, 0))
            plt.savefig("cifar_hex_{}.pdf".format(i),
                        bbox_inches='tight',
                        pad_inches=0)

    else:
        import scipy.misc
        for i, image in enumerate(images[:5]):
            scipy.misc.imsave("cifar_{}.png".format(i),
                              upsample(image.transpose(1, 2, 0), factor=10))


def get_cifar10_data(datadir, trainfn, valfn, hex_sampling=''):
    assert os.path.exists(datadir), 'Datadir does not exist: %s' % datadir

    raw_datadir = os.path.join(datadir, 'cifar-10-batches-py')
    assert os.path.exists(raw_datadir), \
        'Could not find CIFAR10 data. Please download cifar-10 from ' \
        'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz,\n' \
        'Extract the data with: tar zxvf cifar-10-python.tar.gz\n' \
        'And move the cifar-10-batches-py folder to datadir (' + datadir + ').'

    processed_datadir = get_preprocess_folder(datadir, hex_sampling)

    if not os.path.exists(processed_datadir):
        preprocess_cifar10(datadir, hex_sampling=hex_sampling)

    train = np.load(os.path.join(processed_datadir, trainfn))
    val = np.load(os.path.join(processed_datadir, valfn))
    train_data = train['data']
    train_labels = train['labels']
    val_data = val['data']
    val_labels = val['labels']

    return train_data, train_labels, val_data, val_labels


def preprocess_cifar10(datadir, hex_sampling=''):
    print('Preprocessing...')

    # Load batches
    print('   Loading...')

    load_cifar_fc = _load_cifar10_batch

    train_batch_fns = [os.path.join(datadir, 'cifar-10-batches-py',
                                    'data_batch_' + str(i))
                       for i in range(1, 6)]
    train_batches = [load_cifar_fc(fn) for fn in train_batch_fns]
    test_batch = load_cifar_fc(os.path.join(datadir, 'cifar-10-batches-py',
                                            'test_batch'))

    # Stack the batches into one big array
    train_data_all = np.vstack([train_batches[i][0] for i in
                                range(len(train_batches))]).astype('float32')

    train_labels_all = np.vstack([train_batches[i][1] for i in
                                  range(len(train_batches))]).flatten()
    test_data = test_batch[0].astype('float32')
    test_labels = test_batch[1]

    # Create train / val split of full train set
    train_data = train_data_all[:40000]
    train_labels = train_labels_all[:40000]
    val_data = train_data_all[40000:]
    val_labels = train_labels_all[40000:]

    if hex_sampling == '':
        mask = np.ones(train_data_all.shape[-2:], dtype='bool')
    elif hex_sampling == 'axial':
        print('   Hex sampling...')
        train_data_all = _sample_hex_axial(train_data_all)
        train_data = _sample_hex_axial(train_data)
        test_data = _sample_hex_axial(test_data)
        val_data = _sample_hex_axial(val_data)

        ny, nx = train_data.shape[-2:]
        mask = hexa.mask.square_axial(ny, nx) == 1
    else:
        raise ValueError('Unknown option "{}"'.format(hex_sampling))

    print('Saving a few examples...')
    save_images(datadir, list(train_data[:10]),
                hex_sampling=hex_sampling)

    train_data_all_no_padding = train_data_all[..., mask]
    train_data_no_padding = train_data[..., mask]
    val_data_no_padding = val_data[..., mask]
    test_data_no_padding = test_data[..., mask]

    # Contrast normalize
    print('   Normalizing...')
    train_data_all_no_padding = normalize(train_data_all_no_padding)
    train_data_no_padding = normalize(train_data_no_padding)
    val_data_no_padding = normalize(val_data_no_padding)
    test_data_no_padding = normalize(test_data_no_padding)

    print('   Computing whitening matrix...')
    train_data_all_flat = train_data_all_no_padding.reshape(
        train_data_all.shape[0], -1).T
    train_data_flat = train_data_no_padding.reshape(train_data.shape[0], -1).T
    val_data_flat = val_data_no_padding.reshape(val_data.shape[0], -1).T
    test_data_flat = test_data_no_padding.reshape(test_data.shape[0], -1).T
    pca = PCA(D=train_data_flat, n_components=train_data_flat.shape[1])
    pca_all = PCA(
        D=train_data_all_flat, n_components=train_data_all_flat.shape[1])

    print('   Whitening data...')
    train_data_all_flat = pca_all.transform(
                            D=train_data_all_flat, whiten=True, ZCA=True)

    train_data_flat = pca.transform(
                            D=train_data_flat, whiten=True, ZCA=True)
    train_data_all[..., mask] = \
        train_data_all_flat.T.reshape(train_data_all.shape[0:2] + (-1,))

    train_data[..., mask] = \
        train_data_flat.T.reshape(train_data.shape[0:2] + (-1,))

    test_data_flat = pca_all.transform(
                            D=test_data_flat, whiten=True, ZCA=True)
    test_data[..., mask] = \
        test_data_flat.T.reshape(test_data.shape[0:2] + (-1,))

    val_data_flat = pca.transform(
                            D=val_data_flat, whiten=True, ZCA=True)
    val_data[..., mask] = \
        val_data_flat.T.reshape(val_data.shape[0:2] + (-1,))

    # train_data_all[..., mask] = \
    #     global_contrast_normalize(train_data_all_flat).reshape(
    #         train_data_all_no_padding.shape)
    # train_data[..., mask] = global_contrast_normalize(train_data_flat).reshape(
    #         train_data_no_padding.shape)
    # val_data[..., mask] = global_contrast_normalize(val_data_flat).reshape(
    #         val_data_no_padding.shape)
    # test_data[..., mask] = global_contrast_normalize(test_data_flat).reshape(
    #         test_data_no_padding.shape)

    print('   Saving...')
    outputdir = get_preprocess_folder(datadir, hex_sampling)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    np.savez(os.path.join(outputdir, 'train.npz'),
             data=train_data,
             labels=train_labels)
    np.savez(os.path.join(outputdir, 'train_all.npz'),
             data=train_data_all,
             labels=train_labels_all)
    np.savez(os.path.join(outputdir, 'val.npz'),
             data=val_data,
             labels=val_labels)
    np.savez(os.path.join(outputdir, 'test.npz'),
             data=test_data,
             labels=test_labels)

    print('Preprocessing complete')


def _load_cifar10_batch(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict['data'].reshape(-1, 3, 32, 32), dict['labels']


def _sample_hex_axial(data):
    hex_data = np.zeros(
        (data.shape[0:2] + sample_cartesian2hexa(data[0, 0]).shape))

    for image in range(data.shape[0]):
        for channel in range(data.shape[1]):
            hex_data[image, channel, ...] = \
                sample_cartesian2hexa(data[image, channel])

    return hex_data


class PCA(object):

    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        """
        The computation works as follows:
        The covariance is C = 1/(n-1) * D * D.T
        The eigendecomp of C is: C = V Sigma V.T
        Let Y = 1/sqrt(n-1) * D
        Let U S V = svd(Y),
        Then the columns of U are the eigenvectors of:
        Y * Y.T = C
        And the singular values S are the sqrts of the eigenvalues of C
        We can apply PCA by multiplying by U.T
        """

        # We require scaled, zero-mean data to SVD,
        # But we don't want to copy or modify user data
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=False, ZCA=False,
                  regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by Sigma^(-1/2) U.T
        Any orthogonal transformation of this is also white,
        and when ZCA=True we choose:
         U Sigma^(-1/2) U.T
        """
        if whiten:
            # Compute Sigma^(-1/2) = S^-1,
            # with smoothing for numerical stability
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                # The ZCA whitening matrix
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                # The whitening matrix
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)
