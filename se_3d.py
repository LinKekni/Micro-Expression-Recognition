from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D,Reshape, Dense, multiply, add, Permute, Conv3D
import keras.backend as K
import numpy as np

#K.image_data_format() == 'channels_first'
#K.set_image_dim_ordering()='th'
def squeeze_excite_block(input, ratio=8):
    ''' Create a channel-wise squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    print("init=",init)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    #filters = init.shape[channel_axis]
    print("filters=",filters)
    se_shape = (1, 1, 1, filters)

    se = GlobalMaxPooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='VarianceScaling', use_bias=False)(se)#
    
    se = Dense(filters, activation='sigmoid',kernel_initializer='VarianceScaling', use_bias=False)(se)#, kernel_initializer='RandomUniform'
    
    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se) #(4,1,2,3)
    ##se = Permute((1,2,3,4))(se)
    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input):
    ''' Create a spatial squeeze-excite block

    Args:
        input: input tensor

    Returns: a keras tensor

    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    se = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input)

    x = multiply([input, se])
    return x


def channel_spatial_squeeze_excite(input, ratio=16):
    ''' Create a spatial squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x

