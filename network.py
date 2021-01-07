from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, concatenate, Dropout
from keras.models import Model
import keras.backend as K
from keras import optimizers

import info as I

thre_depth = I.thre_depth
thre_diff = I.thre_diff
thre_diff = 1

def BuildUnet(
    num_ch,
    lr,
    rate_dropout,
    is_transfer_learn=False,
    is_transfer_encoder=False,
    shape_patch=I.shape_patch
):
    def EncodeBlock(x, ch):
        def BaseEncode(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate_dropout)(x)
            x = Conv2D(ch, (3, 3), padding='same')(x)
            return x
        
        x = BaseEncode(x)
        x = BaseEncode(x)
        return x

    def DecodeBlock(x, shortcut, ch):
        def BaseDecode(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate_dropout)(x)
            x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
            return x
        
        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, shortcut])

        x = BaseDecode(x)
        x = BaseDecode(x)
        return x
    
    def MSE_Mask(y_true, y_pred):
        diff = y_true[:, :, :, 0]
        depth = y_true[:, :, :, 1]
        # mask = y_true[:, :, :, 2]

        is_depth_valid = depth > thre_depth
        is_depth_close = K.abs(diff) < thre_diff
        is_valid = K.all(K.stack([is_depth_valid, is_depth_close], axis=0), axis=0)
        mask = K.cast(is_valid, 'float32')

        len_mask = K.sum(mask)
        mse = K.sum(K.square(diff - y_pred[:, :, :, 0]) * mask) / len_mask
        return mse

    input_patch = Input(shape=(*shape_patch, num_ch))
    e0 = Conv2D(8, (1, 1), padding='same')(input_patch)
    e0 = Activation('tanh')(e0)

    e0 = EncodeBlock(e0, 16)

    e1 = AveragePooling2D((2, 2))(e0)
    e1 = EncodeBlock(e1, 32)

    e2 = AveragePooling2D((2, 2))(e1)
    e2 = EncodeBlock(e2, 64)

    e3 = AveragePooling2D((2, 2))(e2)
    e3 = EncodeBlock(e3, 128)

    d2 = DecodeBlock(e3, e2, 64)
    d1 = DecodeBlock(d2, e1, 32)
    d0 = DecodeBlock(d1, e0, 16)

    output_patch = Conv2D(1, (1, 1), padding='same')(d0)

    model = Model(input_patch, output_patch)

    # Transfer Learning
    if is_transfer_learn:
        for l in model.layers[:38]:
            l.trainable = False
    elif is_transfer_encoder:
        for l in model.layers[38:]:
            l.trainable = False

    adam = optimizers.Adam(lr=lr)
    model.compile(
        optimizer=adam,
        loss=MSE_Mask
    )
    return model