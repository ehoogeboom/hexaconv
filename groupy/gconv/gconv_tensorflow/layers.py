"""Contains the group equivariant convolutional tensorflow layers."""
import tensorflow as tf
from groupy.gconv.gconv_tensorflow import utils
from groupy.gconv.gconv_tensorflow.transform_kernel import transform_kernel_2d_nhwc
from groupy.gconv import make_gconv_indices as idx
from groupy.hexa import mask


class SplitGConv2D(tf.layers.Layer):
    """Group convolution base class for split plane groups.

    A plane group (aka wallpaper group) is a group of distance-preserving transformations that includes two independent
    discrete translations.

    A group is called split (or symmorphic) if every element in this group can be written as the composition of an
    element from the "stabilizer of the origin" and a translation. The stabilizer of the origin consists of those
    transformations in the group that leave the origin fixed. For example, the stabilizer in the rotation-translation
    group p4 is the set of rotations around the origin, which is (isomorphic to) the group C4.

    Most plane groups are split, but some include glide-reflection generators; such groups are not split.
    For split groups G, the G-conv can be split into a "filter transform" and "translational convolution" part.

    Different subclasses of this class implement the filter transform for various groups, while this class implements
    the common functionality.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    kernel_mask = None
    output_mask = None

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(SplitGConv2D, self).__init__(trainable=trainable, name=name,
                                           activity_regularizer=activity_regularizer,
                                           **kwargs)
        self.filters = filters
        if not isinstance(kernel_size, int):
            raise ValueError('kernel_size must be a integer. Received: ' + str(kernel_size) + ' of type ' + str(type(kernel_size)))
        self.kernel_size = kernel_size
        self.strides = utils.normalize_tuple(strides, 2, 'strides')
        self.padding = utils.normalize_padding(padding)
        if data_format != 'channels_last':
            raise NotImplemented('Currently only channels_last data_format is supported. Received:' + str(data_format))
        self.data_format = utils.convert_data_format(data_format)
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = tf.layers.InputSpec(ndim=4)

    def get_masks(self, input_shape):
        return None, None

    @property
    def input_stabilizer_size(self):
        raise NotImplementedError('Subclasses should implement this!')

    @property
    def output_stabilizer_size(self):
        raise NotImplementedError('Subclasses should implement this!')

    @property
    def transformation_indices(self):
        raise NotImplementedError('Subclasses should implement this!')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.kernel_mask, self.output_mask = self.get_masks(input_shape)
        channel_axis = -1 if self.data_format == 'NHWC' else 1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = input_shape[channel_axis].value
        kernel_shape = (self.input_stabilizer_size, self.kernel_size, self.kernel_size, input_dim // self.input_stabilizer_size, self.filters)

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)

        self.transformed_kernel = transform_kernel_2d_nhwc(
            self.kernel_mask * self.kernel if self.kernel_mask is not None else self.kernel,
            self.transformation_indices)

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = tf.layers.InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        strides = (1,) + self.strides + (1,) if self.data_format == 'NHWC' else (1, 1) + self.strides

        outputs = tf.nn.conv2d(input=inputs,
                               filter=self.transformed_kernel,
                               strides=strides,
                               padding=self.padding.upper(),
                               data_format=self.data_format,
                               name=self.name)

        if self.use_bias:
            if self.data_format != 'NHWC':
                raise NotImplemented('Currently only NHWC data_format is supported. Received:' + str(self.data_format))

            outputs_shape = outputs.get_shape().as_list()
            outputs_flat = tf.reshape(outputs, [-1, self.filters, self.output_stabilizer_size, 1])
            outputs_flat = tf.nn.bias_add(outputs_flat, self.bias, data_format='NCHW')
            outputs = tf.reshape(outputs_flat, [-1] + outputs_shape[1:])

        if self.output_mask is not None:
            outputs *= self.output_mask

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'NHWC':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size,
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=1)
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space + [self.filters * self.output_stabilizer_size])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size,
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=1)
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters * self.output_stabilizer_size] + new_space)


class SplitHexGConv2D(SplitGConv2D):
    """Group convolution base class for P6/P6M groups.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    def get_masks(self, input_shape):
        kernel_mask = tf.convert_to_tensor(mask.hexagon_axial(self.kernel_size)[None, ..., None, None], dtype=self.dtype, name='kernel_mask')
        output_shape = self.compute_output_shape(input_shape).as_list()
        ny, nx = output_shape[1:3] if self.data_format == 'NHWC' else output_shape[-2:]
        output_mask = tf.convert_to_tensor(mask.square_axial(ny, nx)[None, ..., None], dtype=self.dtype, name='output_mask')
        return kernel_mask, output_mask


class P4ConvZ2(SplitGConv2D):
    """Z2 to P4 group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 1

    @property
    def output_stabilizer_size(self):
        return 4

    @property
    def transformation_indices(self):
        return idx.make_c4_z2_indices(ksize=self.kernel_size)


class P4ConvP4(SplitGConv2D):
    """P4 to P4 group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 4

    @property
    def output_stabilizer_size(self):
        return 4

    @property
    def transformation_indices(self):
        return idx.make_c4_p4_indices(ksize=self.kernel_size)


class P4MConvZ2(SplitGConv2D):
    """Z2 to P4M group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 1

    @property
    def output_stabilizer_size(self):
        return 8

    @property
    def transformation_indices(self):
        return idx.make_d4_z2_indices(ksize=self.kernel_size)


class P4MConvP4M(SplitGConv2D):
    """Z4M to P4M group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 8

    @property
    def output_stabilizer_size(self):
        return 8

    @property
    def transformation_indices(self):
        return idx.make_d4_p4m_indices(ksize=self.kernel_size)


class Z2ConvZ2Axial(SplitHexGConv2D):
    """Z2 to Z2 group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 1

    @property
    def output_stabilizer_size(self):
        return 1

    @property
    def transformation_indices(self):
        return idx.make_c6_z2_indices(ksize=self.kernel_size)


class P6ConvZ2Axial(SplitHexGConv2D):
    """Z2 to P6 group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 1

    @property
    def output_stabilizer_size(self):
        return 6

    @property
    def transformation_indices(self):
        return idx.make_c6_z2_indices(ksize=self.kernel_size)


class P6ConvP6Axial(SplitHexGConv2D):
    """P6 to P6 group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 6

    @property
    def output_stabilizer_size(self):
        return 6

    @property
    def transformation_indices(self):
        return idx.make_c6_p6_indices(ksize=self.kernel_size)


class P6MConvZ2Axial(SplitHexGConv2D):
    """Z2 to P6M group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """

    @property
    def input_stabilizer_size(self):
        return 1

    @property
    def output_stabilizer_size(self):
        return 12

    @property
    def transformation_indices(self):
        return idx.make_d6_z2_indices(ksize=self.kernel_size)


class P6MConvP6MAxial(SplitHexGConv2D):
    """P6M to P6M group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
    kernel_size: An integer specifying the length of the convolution window.
    strides: An integer or tuple/list of n integers, specifying the stride length of the convolution.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    activation: Activation function. Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    """
    @property
    def input_stabilizer_size(self):
        return 12

    @property
    def output_stabilizer_size(self):
        return 12

    @property
    def transformation_indices(self):
        return idx.make_d6_p6m_indices(ksize=self.kernel_size)
