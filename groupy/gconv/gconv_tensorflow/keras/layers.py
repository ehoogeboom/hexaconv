"""Contains the group equivariant convolutional Keras layers."""
import tensorflow as tf
import groupy.gconv.gconv_tensorflow.layers as tf_layers


class SplitGConv2D(tf_layers.SplitGConv2D, tf.keras.layers.Layer):
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

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if data_format is None:
            data_format = tf.keras.backend.image_data_format()
        super(SplitGConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=tf.keras.activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.get(kernel_initializer),
            bias_initializer=tf.keras.initializers.get(bias_initializer),
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            kernel_constraint=tf.keras.constraints.get(kernel_constraint),
            bias_constraint=tf.keras.constraints.get(bias_constraint),
            **kwargs)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(SplitGConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SplitHexGConv2D(SplitGConv2D, tf_layers.SplitHexGConv2D):
    """Group convolution base class for P6/P6M groups.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P4ConvZ2(SplitGConv2D, tf_layers.P4ConvZ2):
    """Z2 to P4 group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P4ConvP4(SplitGConv2D, tf_layers.P4ConvP4):
    """P4 to P4 group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P4MConvZ2(SplitGConv2D, tf_layers.P4MConvZ2):
    """Z2 to P4M group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P4MConvP4M(SplitGConv2D, tf_layers.P4MConvP4M):
    """P4M to P4M group convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class Z2ConvZ2Axial(SplitHexGConv2D, tf_layers.Z2ConvZ2Axial):
    """Z2 to Z2 group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P6ConvZ2Axial(SplitHexGConv2D, tf_layers.P6ConvZ2Axial):
    """Z2 to P6 group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P6ConvP6Axial(SplitHexGConv2D, tf_layers.P6ConvP6Axial):
    """P6 to P6 group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P6MConvZ2Axial(SplitHexGConv2D, tf_layers.P6MConvZ2Axial):
    """Z2 to P6M group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass


class P6MConvP6MAxial(SplitHexGConv2D, tf_layers.P6MConvP6MAxial):
    """P6M to P6M group convolution layer on a hexagonal grid using axial coordinates.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis), e.g. `input_shape=(128, 128, 3)` for
    128x128 RGB pictures in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
      kernel_size: An integer specifying the width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    """

    pass
