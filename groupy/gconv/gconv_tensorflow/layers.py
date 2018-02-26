import tensorflow as tf
from groupy.gconv.gconv_tensorflow import utils
from groupy.gconv.gconv_tensorflow.transform_filter import transform_filter_2d_nhwc
from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices,\
    make_d4_z2_indices, make_d4_p4m_indices, flatten_indices


class SplitGConv2D(tf.layers.Layer):
    # TODO: Conform with tf.layers and use data_format={channels_last, channels_first}
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='NHWC',
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
            # TODO: Also handle tuples and check if non square kernels are supported.
            raise ValueError('kernel_size must be a integer. Received: ' + str(kernel_size) + ' of type ' + str(type(kernel_size)))
        self.kernel_size = kernel_size
        self.strides = utils.normalize_tuple(strides, 2, 'strides')
        self.padding = utils.normalize_padding(padding)
        if data_format != 'NHWC':
            raise NotImplemented('Currently only NHWC data_format is supported. Received:' + str(data_format))
        self.data_format = data_format
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = tf.layers.InputSpec(ndim=4)
    
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
        channel_axis = -1 if self.data_format == 'NHWC' else 1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = input_shape[channel_axis].value
        kernel_shape = (self.kernel_size, self.kernel_size, input_dim, self.filters)
        
        gconv_shape_info = (self.filters, self.output_stabilizer_size, input_dim // self.input_stabilizer_size, self.input_stabilizer_size, self.kernel_size)

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
                                        
        self.transformed_kernel = transform_filter_2d_nhwc(w=self.kernel, flat_indices=self.transformation_indices, shape_info=gconv_shape_info)
        
        if self.use_bias:
            raise NotImplemented('Bias not supported yet!')
            #self.bias = self.add_variable(name='bias',
                                          #shape=(self.filters,),
                                          #initializer=self.bias_initializer,
                                          #regularizer=self.bias_regularizer,
                                          #constraint=self.bias_constraint,
                                          #trainable=True,
                                          #dtype=self.dtype)
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
            raise NotImplemented('Bias not supported yet!')
          #if self.data_format == 'channels_first':
            #if self.rank == 1:
              ## nn.bias_add does not accept a 1D input tensor.
              #bias = array_ops.reshape(self.bias, (1, self.filters, 1))
              #outputs += bias
            #if self.rank == 2:
              #outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            #if self.rank == 3:
              ## As of Mar 2017, direct addition is significantly slower than
              ## bias_add when computing gradients. To use bias_add, we collapse Z
              ## and Y into a single dimension to obtain a 4D input tensor.
              #outputs_shape = outputs.shape.as_list()
              #outputs_4d = array_ops.reshape(outputs,
                                             #[outputs_shape[0], outputs_shape[1],
                                              #outputs_shape[2] * outputs_shape[3],
                                              #outputs_shape[4]])
              #outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
              #outputs = array_ops.reshape(outputs_4d, outputs_shape)
          #else:
            #outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

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
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space + [self.filters * self.output_stabilizer_size])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters * self.output_stabilizer_size] + new_space)


class P4ConvZ2(SplitGConv2D):
    @property
    def input_stabilizer_size(self):
        return 1
    
    @property
    def output_stabilizer_size(self):
        return 4
    
    @property    
    def transformation_indices(self):
        return flatten_indices(make_c4_z2_indices(ksize=self.kernel_size))
        

class P4ConvP4(SplitGConv2D):        
    @property
    def input_stabilizer_size(self):
        return 4

    @property
    def output_stabilizer_size(self):
        return 4
    
    @property    
    def transformation_indices(self):
        return flatten_indices(make_c4_p4_indices(ksize=self.kernel_size))


class P4MConvZ2(SplitGConv2D):
    @property
    def input_stabilizer_size(self):
        return 1
    
    @property
    def output_stabilizer_size(self):
        return 8
    
    @property    
    def transformation_indices(self):
        return flatten_indices(make_d4_z2_indices(ksize=self.kernel_size))
    

class P4MConvP4M(SplitGConv2D):
    @property
    def input_stabilizer_size(self):
        return 8
    
    @property
    def output_stabilizer_size(self):
        return 8
    
    @property    
    def transformation_indices(self):
        return flatten_indices(make_d4_p4m_indices(ksize=self.kernel_size))