import os

import numpy as np
from groupy.hexa.hexa_sample import sample_cartesian2hexa
from groupy import hexa
import sys
from os.path import join, isdir
import scipy.misc


def normalize(data, eps=1e-8):
    data -= data.mean(axis=(1, 2), keepdims=True)
    std = np.sqrt(data.var(axis=(1, 2), ddof=1, keepdims=True))
    std[std < eps] = 1.
    data /= std
    return data


def get_preprocess_folder(datadir, hex_sampling, seed):
    if hex_sampling == '':
        path = os.path.join(datadir, 'preprocessed')
    elif hex_sampling == 'axial':
        path = os.path.join(datadir, 'preprocessed_hex_axial')
    else:
        raise ValueError('Unknown option "{}"'.format(hex_sampling))

    return path + '_seed' + str(seed)


def get_aid_data(datadir, trainfn, valfn, seed=0, hex_sampling=''):
    assert os.path.exists(datadir), 'Datadir does not exist: %s' % datadir

    processed_datadir = get_preprocess_folder(datadir, hex_sampling, seed)

    if not os.path.exists(processed_datadir):
        preprocess_aid(datadir, hex_sampling=hex_sampling, seed=seed)

    train = np.load(os.path.join(processed_datadir, trainfn))
    val = np.load(os.path.join(processed_datadir, valfn))
    train_data = train['data']
    train_labels = train['labels']
    val_data = val['data']
    val_labels = val['labels']

    return train_data, train_labels, val_data, val_labels


def split(all_images, fraction=0.5, seed=None):
    if seed:
        np.random.seed(seed)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for c, image_class in enumerate(all_images):
        n = len(image_class)

        if n <= 0:
            continue

        selection = np.random.choice(n, size=int(fraction * n), replace=False)

        for i, image in enumerate(image_class):
            if i in selection:
                train_data.append(image)
                train_labels.append(c)

            else:
                test_data.append(image)
                test_labels.append(c)

    max_c = max(max(test_labels), max(train_labels))
    assert max_c == 29, \
        'There are only 30 classes, not {}'.format(max_c)

    train_data = np.array(train_data, dtype='float32')
    labels_train = np.array(train_labels, dtype='int32')
    test_data = np.array(test_data, dtype='float32')
    labels_test = np.array(test_labels, dtype='int32')

    return train_data, labels_train, test_data, labels_test


def preprocess_aid(datadir, hex_sampling, seed):
    print 'Preprocessing...'

    # Load batches
    print '   Loading...'

    paths = [[join(datadir, path, i) for i in os.listdir(join(datadir, path))
              if len(i) > 4 and i[-4:] == '.jpg']
             for path in os.listdir(datadir)
             if isdir(join(datadir, path))]

    print '   Loading images...'
    all_images = [[scipy.misc.imresize(
                        scipy.misc.imread(image), (64, 64)).transpose(2, 0, 1)
                   for image in path]
                  for path in paths]

    # Remove empty paths
    for i in range(len(all_images)-1, -1, -1):
        if len(all_images[i]) == 0:
            all_images.pop(i)

    train_data_all, train_labels_all, test_data, test_labels = \
        split(all_images, fraction=0.8, seed=seed)

    selection = np.random.choice(len(train_data_all), size=10, replace=False)

    if hex_sampling == '':
        mask = np.ones(train_data_all.shape[-2:], dtype='bool')
    elif hex_sampling == 'axial':
        print '   Hex sampling...'
        train_data_all = _sample_hex_axial(train_data_all)
        test_data = _sample_hex_axial(test_data)

        ny, nx = train_data_all.shape[-2:]
        mask = hexa.mask.square_axial(ny, nx) == 1
    else:
        raise ValueError('Unknown option "{}"'.format(hex_sampling))

    train_data_all_no_padding = train_data_all[..., mask]
    test_data_no_padding = test_data[..., mask]

    # Contrast normalize
    print '   Normalizing...'
    train_data_all_no_padding = normalize(train_data_all_no_padding)
    test_data_no_padding = normalize(test_data_no_padding)

    train_data_all[..., mask] = train_data_all_no_padding

    test_data[..., mask] = test_data_no_padding

    print '   Saving...'
    outputdir = get_preprocess_folder(datadir, hex_sampling, seed)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    np.savez(os.path.join(outputdir, 'train_all.npz'),
             data=train_data_all,
             labels=train_labels_all)
    np.savez(os.path.join(outputdir, 'test.npz'),
             data=test_data,
             labels=test_labels)

    print 'Preprocessing complete'


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
