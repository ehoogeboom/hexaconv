import tensorflow as tf
import groupy.gconv.gconv_tensorflow.layers as tf_layers


class SplitGConv2D(tf_layers.SplitGConv2D, tf.keras.layers.Layer):
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
    pass


class P4ConvZ2(SplitGConv2D, tf_layers.P4ConvZ2):
    pass


class P4ConvP4(SplitGConv2D, tf_layers.P4ConvP4):
    pass


class P4MConvZ2(SplitGConv2D, tf_layers.P4MConvZ2):
    pass


class P4MConvP4M(SplitGConv2D, tf_layers.P4MConvP4M):
    pass


class Z2ConvZ2Axial(SplitHexGConv2D, tf_layers.Z2ConvZ2Axial):
    pass


class P6ConvZ2Axial(SplitHexGConv2D, tf_layers.P6ConvZ2Axial):
    pass


class P6ConvP6Axial(SplitHexGConv2D, tf_layers.P6ConvP6Axial):
    pass


class P6MConvZ2Axial(SplitHexGConv2D, tf_layers.P6MConvZ2Axial):
    pass


class P6MConvP6MAxial(SplitHexGConv2D, tf_layers.P6MConvP6MAxial):
    pass
