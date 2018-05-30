import pytest
from groupy.gconv.gconv_tensorflow import utils


def test_convert_data_format():
    assert utils.convert_data_format('channels_first') == 'NCHW'
    assert utils.convert_data_format('channels_last') == 'NHWC'

    with pytest.raises(ValueError):
        utils.convert_data_format('invalid',)


def test_normalize_tuple():
    assert utils.normalize_tuple(2, n=3, name='strides') == (2, 2, 2)
    assert utils.normalize_tuple((2, 1, 2), n=3, name='strides') == (2, 1, 2)

    with pytest.raises(ValueError):
        utils.normalize_tuple((2, 1), n=3, name='strides')

    with pytest.raises(ValueError):
        utils.normalize_tuple(None, n=3, name='strides')


def test_normalize_padding():
    assert utils.normalize_padding('SAME') == 'same'
    assert utils.normalize_padding('VALID') == 'valid'

    with pytest.raises(ValueError):
        utils.normalize_padding('invalid')


def test_conv_output_length():
    assert utils.conv_output_length(4, 2, 'same', 1, 1) == 4
    assert utils.conv_output_length(4, 2, 'same', 2, 1) == 2
    assert utils.conv_output_length(4, 2, 'valid', 1, 1) == 3
    assert utils.conv_output_length(4, 2, 'valid', 2, 1) == 2
    assert utils.conv_output_length(4, 2, 'full', 1, 1) == 5
    assert utils.conv_output_length(4, 2, 'full', 2, 1) == 3
    assert utils.conv_output_length(5, 2, 'valid', 2, 2) == 2
