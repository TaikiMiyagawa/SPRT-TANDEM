""" ResNet version 1 and 2 implementation with TF2.
    Code modified from the official TF1 code.
    References: 
        He et al., 2016. "Deep Residual Learning for Image Recognition."
        He et al., 2016. "Identity Mappings in Deep Residual Networks."
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# ResNet util functions
################################################################################
def _get_block_sizes_v1(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
        resnet_size: The number of convolutional layers needed in the model. 6n+2.
    Returns:
        A list of block sizes to use in building the model.
    Raises:
        KeyError: if invalid resnet_size is received.
    Remarks:
        Used in gt_ressize_dependent_params_v1.
    """
    choices = {
        8: [1, 1, 1], # 0.09M, (16, 32, 64)
        14: [2, 2, 2], # 0.18M
        20: [3, 3, 3], # 0.25M
        32: [5, 5, 5], # 0.46M
        44: [7, 7, 7], # 0.66M
        56: [9, 9, 9], # 0.85M
        110: [18, 18, 18], # 1.7M
        218: [36, 36, 36] # 3.4M
    }

    try:
        return choices[resnet_size]
        
    except KeyError:
        err = ('Could not find layers for selected Resnet v1 size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, choices.keys()))
        raise ValueError(err)


def _get_ressize_dependent_params_v1(resnet_size):
    """
    Arg:
        resnet_size: An integer. 6n+2.
    Returns:
        bottleneck: A boolean. Use regular blocks or bottleneck blocks.
        kernel_size: The kernel size to use for the initial convolution.
        conv_stride: stride size for the initial convolution
        first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
        first_pool_stride: stride size for the first pooling layer.
            Not used if first_pool_size is None.
        block_sizes: A list containing n values, where n is the number of
            sets of block layers desired. Each value should be 
            the number of blocks in the i-th set.
        block_strides: List of integers representing the desired 
            stride size for each of the sets of block layers. 
            Should be same length as block_sizes.
        final_size: The expected size of the model after the second pooling.
    """
    dict_resparams = dict()
    
    dict_resparams["bottleneck"] = False
    dict_resparams["final_size"] = 64
    dict_resparams["kernel_size"] = 3
    dict_resparams["conv_stride"] = 1
    dict_resparams["first_pool_size"] = None
    dict_resparams["first_pool_stride"] = None
    dict_resparams["block_sizes"] = _get_block_sizes_v1(resnet_size)
    dict_resparams["block_strides"] =[1, 2, 2]

    return dict_resparams


def _get_block_sizes_v2(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
        resnet_size: The number of convolutional layers needed in the model.
    Returns:
        A list of block sizes to use in building the model.
    Raises:
        KeyError: if invalid resnet_size is received.
    Remarks:
        Used in gt_ressize_dependent_params_v2.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3], 
        152: [3, 8, 36, 3], 
        200: [3, 24, 36, 3], 
    }

    try:
        return choices[resnet_size]
        
    except KeyError:
        err = ('Could not find layers for selected Resnet v2 size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, choices.keys()))
        raise ValueError(err)


def _get_ressize_dependent_params_v2(resnet_size):
    """
    Arg:
        resnet_size: An integer.
    Returns:
        bottleneck: A boolean. Use regular blocks or bottleneck blocks.
        kernel_size: The kernel size to use for the initial convolution.
        conv_stride: stride size for the initial convolution
        first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
        first_pool_stride: stride size for the first pooling layer.
            Not used if first_pool_size is None.
        block_sizes: A list containing n values, where n is the number of
            sets of block layers desired. Each value should be 
            the number of blocks in the i-th set.
        block_strides: List of integers representing the desired 
            stride size for each of the sets of block layers. 
            Should be same length as block_sizes.
        final_size: The expected size of the model after the second pooling.

    """
    dict_resparams = dict()
    
    if resnet_size < 50:
        dict_resparams["bottleneck"] = False
        dict_resparams["final_size"] = 512 

    else:
        dict_resparams["bottleneck"] = True
        dict_resparams["final_size"] = 2048 

    dict_resparams["block_sizes"] = _get_block_sizes_v2(resnet_size)
    dict_resparams["kernel_size"] = 7
    dict_resparams["conv_stride"] = 2
    dict_resparams["first_pool_size"] = 3
    dict_resparams["first_pool_stride"] = 2
    dict_resparams["block_strides"] =[2, 2, 2, 2]

    return dict_resparams


def get_ressize_dependent_params(resnet_version, resnet_size):
    """ Retrieve bottleneck flag, final size, 
        and the size of each block layer in the ResNet model.
    Args:
        resnet_version: An int, 1 or 2.
        resnet_size: An int.
    Returns:
        bottleneck: A boolean. Use regular blocks or bottleneck blocks.
        kernel_size: The kernel size to use for the initial convolution.
        conv_stride: stride size for the initial convolution
        first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
        first_pool_stride: stride size for the first pooling layer.
            Not used if first_pool_size is None.
        block_sizes: A list containing n values, where n is the number of
            sets of block layers desired. Each value should be 
            the number of blocks in the i-th set.
        block_strides: List of integers representing the desired 
            stride size for each of the sets of block layers. 
            Should be same length as block_sizes.
        final_size: The expected size of the model after the second pooling.
    """
    if resnet_version == 1:
        dict_resparams = _get_ressize_dependent_params_v1(resnet_size)

    elif resnet_version == 2:
        dict_resparams = _get_ressize_dependent_params_v2(resnet_size)

    else:
        raise ValueError('Resnet version should be 1 or 2.')

    return dict_resparams


###############################################################################
# Convenience functions for building the ResNet model.
###############################################################################
def get_batch_norm_layer(data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops

    bn_layer = tf.keras.layers.BatchNormalization(
        axis=1 if data_format == 'channels_first' else 3, 
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, 
        center=True, scale=True,
        fused=True)

    return bn_layer


# conv2d_fixed_padding
class Conv2dFixedPaddingLayer(tf.keras.Model):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

    def __init__(self, filters, kernel_size, strides, data_format):
        """
        filters: Integer, the dimensionality of the output space 
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single 
            integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the
            strides of the convolution along the height and width. 
            Can be a single integer to specify the same value for all
            spatial dimensions. Specifying any stride value != 1 is
            incompatible with specifying any dilation_rate value != 1.
        data_format: A string, one of channels_last (default) or channels_first.
            The ordering of the dimensions in the inputs. channels_last
            corresponds to inputs with shape (batch, height, width, channels) 
            while channels_first corresponds to inputs with shape 
            (batch, channels, height, width). It defaults to the 
            image_data_format value found in your Keras config file at
            ~/.keras/keras.json. If you never set it, then it will be
            "channels_last".
        """
        super(Conv2dFixedPaddingLayer, self).__init__(name="conv2d_fixed_padding")
        self.filters = filters 
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format

        self.fixed_padding_layer = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, 
            strides=self.strides, use_bias=False, 
            padding=('SAME' if self.strides == 1 else 'VALID'), 
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            data_format=self.data_format)

    def fixed_padding(self, inputs):
        """Pads the input along the spatial dimensions independently of input size.
        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        Returns:
            A tensor with the same format as the input with the data either intact
            (if kernel_size == 1) or padded (if kernel_size > 1).
        Remarks:
            self.kernel_size: The kernel to be used in the conv2d or max_pool2d
            operation. Should be a positive integer.
        """
        pad_total = self.kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if self.data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def call(self, inputs):
        if self.strides > 1:
            inputs = self.fixed_padding(inputs)

        inputs = self.fixed_padding_layer(inputs)

        return inputs


################################################################################
# ResNet block definitions.
################################################################################
# building_block_v1
class BuildingBlockV1(tf.keras.Model): 
    """A single block for ResNet v1, without a bottleneck.
    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    """
    def __init__(self, filters, projection_shortcut_layer, strides, data_format):
        """
        Args:
            filters: The number of filters for the convolutions.
            projection_shortcut_layer: The tf.keras.Layer to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
                downsample the input.
            data_format: The input format ('channels_last' or 'channels_first').
        """
        super(BuildingBlockV1, self).__init__(name="building_block_v1")
        self.filters = filters 
        self.projection_shortcut_layer = projection_shortcut_layer
        self.strides = strides
        self.data_format = data_format

        # Construct layers
        if projection_shortcut_layer is not None:
            self.bn0 = get_batch_norm_layer(data_format=data_format)

        self.fixpad1 = Conv2dFixedPaddingLayer(
            filters=filters, kernel_size=3, strides=strides, data_format=data_format)
        self.bn1 = get_batch_norm_layer(data_format)
        self.act1 = tf.keras.layers.Activation("relu")

        self.fixpad2 = Conv2dFixedPaddingLayer(
            filters=filters, kernel_size=3, strides=1,
            data_format=data_format)
        self.bn2 = get_batch_norm_layer(data_format)
        self.act2 = tf.keras.layers.Activation("relu")

    def call(self, inputs, training):
        """
        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference
                mode. Needed for batch normalization.
        Returns:
            The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs

        if self.projection_shortcut_layer is not None:
            shortcut = self.projection_shortcut_layer(inputs)
            shortcut = self.bn0(inputs=shortcut, training=training)

        inputs = self.fixpad1(inputs=inputs)
        inputs = self.bn1(inputs, training=training)
        inputs = self.act1(inputs)

        inputs = self.fixpad2(inputs=inputs)
        inputs = self.bn2(inputs, training=training)
        inputs += shortcut
        inputs = self.act2(inputs)

        return inputs


# building_block_v2
class BuildingBlockV2(tf.keras.Model):
    """A single block for ResNet v2, without a bottleneck.
    Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """
    def __init__(self, filters, projection_shortcut_layer, strides, data_format):
        """
        Args:
            filters: The number of filters for the convolutions.
            projection_shortcut_layer: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
                downsample the input.
            data_format: The input format ('channels_last' or 'channels_first').
        """
        super(BuildingBlockV2, self).__init__(name="building_block_v2")
        self.filters = filters 
        self.projection_shortcut_layer = projection_shortcut_layer
        self.strides = strides
        self.data_format = data_format

        # Construct layers
        self.bn1 = get_batch_norm_layer(data_format)
        self.act1 = tf.keras.layers.Activation("relu")
        self.fixpad1 = Conv2dFixedPaddingLayer(
            filters=filters, kernel_size=3, strides=strides, data_format=data_format)

        self.bn2 = get_batch_norm_layer(data_format)
        self.act2 = tf.keras.layers.Activation("relu")
        self.fixpad2 = Conv2dFixedPaddingLayer(
            filters=filters, kernel_size=3, strides=1, data_format=data_format)

    def call(self, inputs, training):
        """
        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.        
            training: A Boolean for whether the model is in training or inference
                mode. Needed for batch normalization.
        Returns:
            The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs

        inputs = self.bn1(inputs, training=training)
        inputs = self.act1(inputs)
        if self.projection_shortcut_layer is not None:
            shortcut = self.projection_shortcut_layer(inputs)
        inputs = self.fixpad1(inputs)

        inputs = self.bn2(inputs, training=training)
        inputs = self.act2(inputs)
        inputs = self.fixpad2(inputs)

        return inputs + shortcut


# bottleneck_block_v1
class BottleneckBlockV1(tf.keras.Model):
    """A single block for ResNet v1, with a bottleneck.
    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
        Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    """
    def __init__(self, filters, projection_shortcut_layer, strides, data_format):
        """
        Args:
            filters: The number of filters for the convolutions.
            projection_shortcut_layer: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
                downsample the input.
            data_format: The input format ('channels_last' or 'channels_first').
        """
        super(BottleneckBlockV1, self).__init__(name="bottleneck_block_V1")
        self.filters = filters 
        self.projection_shortcut_layer = projection_shortcut_layer
        self.strides = strides
        self.data_format = data_format

        # Construct layers
        if projection_shortcut_layer is not None:
            self.bn0 = get_batch_norm_layer(data_format)

        self.fixpad1 = Conv2dFixedPaddingLayer(
            filters=filters//4, kernel_size=1, strides=1, data_format=data_format)
        self.bn1 = get_batch_norm_layer(data_format)
        self.act1 = tf.keras.layers.Activation("relu")

        self.fixpad2 = Conv2dFixedPaddingLayer(
            filters=filters//4, kernel_size=3, strides=strides, data_format=data_format)
        self.bn2 = get_batch_norm_layer(data_format)
        self.act2 = tf.keras.layers.Activation("relu")

        self.fixpad3 = Conv2dFixedPaddingLayer(
            filters=filters, kernel_size=1, strides=1, data_format=data_format)
        self.bn3 = get_batch_norm_layer(data_format)
        self.act3 = tf.keras.layers.Activation("relu")

    def call(self, inputs, training):
        """
        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.        
            training: A Boolean for whether the model is in training or inference
                mode. Needed for batch normalization.
        Returns:
            The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs

        if self.projection_shortcut_layer is not None:
            shortcut = self.projection_shortcut_layer(inputs)
            shortcut = self.bn0(shortcut, training=training)

        inputs = self.fixpad1(inputs)
        inputs = self.bn1(inputs, training=training)
        inputs = self.act1(inputs)

        inputs = self.fixpad2(inputs)
        inputs = self.bn2(inputs, training=training)
        inputs = self.act2(inputs)

        inputs = self.fixpad3(inputs)
        inputs = self.bn3(inputs, training=training)
        inputs += shortcut
        inputs = self.act3(inputs)

        return inputs


# bottleneck_block_v2
class BottleneckBlockV2(tf.keras.Model):
    """A single block for ResNet v2, without a bottleneck.
    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
        Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Adapted to the ordering conventions of:
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """
    def __init__(self, filters, projection_shortcut_layer, strides, data_format):
        """
        Args:
            filters: The number of filters for the convolutions.
            projection_shortcut_layer: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
                downsample the input.
            data_format: The input format ('channels_last' or 'channels_first').
        """
        super(BottleneckBlockV2, self).__init__(name="bottleneck_block_v2")

        self.filters = filters 
        self.projection_shortcut_layer = projection_shortcut_layer
        self.strides = strides
        self.data_format = data_format

        # Construct layers
        self.bn1 = get_batch_norm_layer(data_format)
        self.act1 = tf.keras.layers.Activation("relu")
        self.fixpad1 = Conv2dFixedPaddingLayer(
            filters=filters//4, kernel_size=1, strides=1, data_format=data_format)

        self.bn2 = get_batch_norm_layer(data_format)
        self.act2 = tf.keras.layers.Activation("relu")
        self.fixpad2 = Conv2dFixedPaddingLayer(
            filters=filters//4, kernel_size=3, strides=strides, data_format=data_format)

        self.bn3 = get_batch_norm_layer(data_format)
        self.act3 = tf.keras.layers.Activation("relu")
        self.fixpad3 = Conv2dFixedPaddingLayer(
            filters=filters, kernel_size=1, strides=1, data_format=data_format)

    def call(self, inputs, training):
        """
        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.        
            training: A Boolean for whether the model is in training or inference
                mode. Needed for batch normalization.
        Returns:
            The output tensor of the block; shape should match inputs.
        Remarks:
            The projection shortcut should come after the first batch norm and ReLU
                since it performs a 1x1 convolution.
        """

        shortcut = inputs

        inputs = self.bn1(inputs, training=training)
        inputs = self.act1(inputs)
        if self.projection_shortcut_layer is not None:
            shortcut = self.projection_shortcut_layer(inputs)
        inputs = self.fixpad1(inputs)

        inputs = self.bn2(inputs, training=training)
        inputs = self.act2(inputs)
        inputs = self.fixpad2(inputs)

        inputs = self.bn3(inputs, training=training)
        inputs = self.act3(inputs)
        inputs = self.fixpad3(inputs)

        return inputs + shortcut


# block_layer
class BlockLayer(tf.keras.Model):
    """Creates one layer of blocks for the ResNet model."""
    def __init__(self, filters, bottleneck, block_sublayer, blocks, strides,
                 name, data_format):
        """
        Args:
            filters: The number of filters for the first convolution of the layer.
            bottleneck: Is the block created a bottleneck block.
            block_sublayer: The block to use within the model, either `BuildingBlock or
                `BottleneckBlock`.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If
                greater than 1, this layer will ultimately downsample the input.
            training: Either True or False, whether we are currently training the
                model. Needed for batch norm.
            name: A string name for the tensor output of the block layer.
            data_format: The input format ('channels_last' or 'channels_first').
        Returns:
            The output tensor of the block layer.
        Remarks:
            Bottleneck blocks end with 4x the number of filters as they start with
                filters_out = filters * 4 if bottleneck else filters
        """
        super(BlockLayer, self).__init__(name="block_layer")

        self.filters = filters
        self.bottleneck = bottleneck
        self.block_sublayer = block_sublayer
        self.blocks = blocks
        self.strides = strides
        self.name_ = name
        self.data_format = data_format
        self.bs = [None]*blocks

        # Construct layers
        self.projection_shortcut_layer = Conv2dFixedPaddingLayer(
                filters=self.filters, kernel_size=1, strides=self.strides,
                data_format=self.data_format)

        self.bs[0] = block_sublayer(
            filters, self.projection_shortcut_layer, strides, data_format)

        for iter_idx in range(1, blocks):
            self.bs[iter_idx] = block_sublayer(filters, None, 1, data_format)

    def call(self, inputs, training):
        """
        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.
        Remarks:
            Only the first block per block_layer uses projection_shortcut_layer and strides
        """
        # With a projection shortcut
        inputs = self.bs[0](inputs, training)

        # Without any projection shortcut
        for iter_idx in range(1, self.blocks):
            inputs = self.bs[iter_idx](inputs, training)

        return tf.identity(inputs, self.name_)


################################################################################
# ResNet definition.
################################################################################
class ResNetModel(tf.keras.Model):
    """Base class for building the Resnet Model."""
    def __init__(
        self, 
        resnet_size,
        bottleneck,
        num_classes,
        kernel_size,
        conv_stride, 
        first_pool_size,
        first_pool_stride,
        block_sizes, 
        block_strides,
        final_size,
        resnet_version=2, 
        data_format=None,
        dtype=tf.float32):
        """
        Args:
            resnet_size: A single integer for the size of the ResNet model.
            bottleneck: A boolean. Use regular blocks or bottleneck blocks.
            num_classes: The number of classes used as labels.
            kernel_size: The kernel size to use for the initial convolution.
            conv_stride: stride size for the initial convolution
            first_pool_size: Pool size to be used for the first pooling layer.
                If none, the first pooling layer is skipped.
            first_pool_stride: stride size for the first pooling layer.
                Not used if first_pool_size is None.
            block_sizes: A list containing n values, where n is the number of
                sets of block layers desired. Each value should be 
                the number of blocks in the i-th set.
            block_strides: List of integers representing the desired 
                stride size for each of the sets of block layers. 
                Should be same length as block_sizes.
            final_size: The expected size of the model after the second pooling.
            resnet_version: Integer representing which version of the ResNet network
                to use. See README for details. Valid values: [1, 2]
            data_format: Input format ('channels_last', 'channels_first', or None).
                If set to None, the format is dependent on whether a GPU is available
                (GPU availabel -> convert to channels-first; otherwise not)
                and the input feature must be channels-last!!
            dtype: The TensorFlow dtype to use for calculations. If not specified
                tf.float32 is used.
        Raises:
            ValueError: if invalid version is selected.
        """
        super(ResNetModel, self).__init__(name="ResNet")

        # ResNet version check
        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise ValueError(
                'Resnet version should be 1 or 2.')

        if resnet_version == 1:
            assert first_pool_size is None,\
                "If resnet_version = 1, first_pool_size is None according to the original paper."

        # Initial checks: dtype and channle format
        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.forced_channels_first = False
        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
            self.forced_channels_first = True

        # Params
        self.data_format = data_format
        self.num_classes = num_classes
        self.dtype_ = dtype

        self.bottleneck = bottleneck
        self.resnet_size = resnet_size

        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides

        self.final_size = final_size
        self.num_filters =\
            final_size // 8 if resnet_version == 2 else final_size // 4
        self.pre_activation = resnet_version == 2
        self.bl = [None] * len(block_sizes)

        if bottleneck:
            if resnet_version == 1:
                self.block_sublayer = BottleneckBlockV1 # not standard
            else:
                self.block_sublayer = BottleneckBlockV2
        else:
            if resnet_version == 1:
                self.block_sublayer = BuildingBlockV1
            else:
                self.block_sublayer = BuildingBlockV2

        # Construct layers
        #####################################################
        self.fixpad1 = Conv2dFixedPaddingLayer(
            filters=self.num_filters, kernel_size=kernel_size,
            strides=conv_stride, data_format=data_format)

        if resnet_version == 1:
            self.bn0 = get_batch_norm_layer(data_format)
            self.act0 = tf.keras.layers.Activation("relu")

        if first_pool_size:
            self.maxpool = tf.keras.layers.MaxPool2D(
                pool_size=first_pool_size, strides=first_pool_stride, 
                padding='SAME', data_format=data_format
            )

        for i, num_blocks in enumerate(block_sizes):
            num_filters = self.num_filters * (2 ** i)
            self.bl[i] = BlockLayer(
                filters=num_filters, 
                bottleneck=bottleneck,
                block_sublayer=self.block_sublayer, 
                blocks=num_blocks,
                strides=block_strides[i],
                name='block_layer{}'.format(i + 1),
                data_format=self.data_format)

        if self.pre_activation:
            self.bn1 = get_batch_norm_layer(data_format)
            self.act1 = tf.keras.layers.Activation("relu") 

        self.dense = tf.keras.layers.Dense(num_classes)


    def call(self, inputs, training):
        """
        Args:
        inputs: A Tensor representing a batch of input images.
        training: A boolean. Set to True to add operations required only when
            training the classifier.
        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        Remarks:
            - We do not include batch normalization or activation functions
              in V2 for the initial conv1 because the first ResNet unit will
              perform these for both the shortcut and non-shortcut paths as 
              part of the firstblock's projection.
            - Convert the inputs from channels_last (NHWC) to channels_first (NCHW). 
              This provides a large performance boost on GPU. See
              https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        # Check data format
        if self.forced_channels_first:
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # 1st stage 
        inputs = self.fixpad1(inputs)
        inputs = tf.identity(inputs, 'initial_conv')

        if self.resnet_version == 1:
            inputs = self.bn0(inputs, training=training)
            inputs = self.act0(inputs)

        if self.first_pool_size:
            inputs = self.maxpool(inputs)
            inputs = tf.identity(inputs, 'initial_max_pool')

        # Remainig stages
        for i, _ in enumerate(self.block_sizes):
            inputs = self.bl[i](inputs=inputs, training=training)

        # Only apply the BN and ReLU for model that does pre_activation in each
        # building/bottleneck block, eg resnet V2.
        if self.pre_activation:
            inputs = self.bn1(inputs, training=training)
            inputs = self.act1(inputs)

        # Global average pooling.
        # The current top layer has shape
        # `batch_size x pool_size x pool_size x final_size`.
        # ResNet does an Average Pooling layer over pool_size,
        # but that is the same as doing a reduce_mean. We do a reduce_mean
        # here because it performs better than AveragePooling2D.
        axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        inputs = tf.reduce_mean(inputs, axes, keepdims=True)
        inputs = tf.identity(inputs, 'final_reduce_mean')

        inputs = tf.reshape(inputs, [-1, self.final_size])
        bottleneck_feat = inputs
        inputs = self.dense(inputs)
        inputs = tf.identity(inputs, 'final_dense') # logits

        return inputs, bottleneck_feat
