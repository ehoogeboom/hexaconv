def convert_data_format(data_format):
    if data_format == 'channels_last':
        return 'NHWC'
    elif data_format == 'channels_first':
        return 'NCHW'
    raise ValueError('Invalid data_format:', data_format)


def normalize_padding(value):
    padding = value.lower()
    if padding not in {'valid', 'same'}:
        raise ValueError('The `padding` argument must be one of "valid", "same". Received: ' + str(padding))
    return padding


def normalize_tuple(value, n, name):
    """Transforms a single integer or iterable of integers into an integer tuple.
    Arguments:
        value: The value to validate and convert. Could an int, or any iterable
            of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. "strides" or
            "kernel_size". This is only used to format error messages.
    Returns:
        A tuple of n integers.
    Raises:
        ValueError: If something else than an int/long or iterable thereof was
            passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
        return value_tuple


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    Arguments:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.
    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride
