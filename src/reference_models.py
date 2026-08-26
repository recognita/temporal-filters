"""
Reference EEG architectures (EEGNet family, DeepConvNet, ShallowConvNet)
used in TemporalFilter.ipynb to sanity-check the learnable filter layers
inside real, published EEG model architectures.

Moved out of the main notebook into its own module: these are unmodified
reference implementations (Lawhern et al., EEGNet), not original work, so
keeping them separate from `temporal_filters.py` makes clear which code is
yours. Removed the unused Keras import block that was left over in the
original notebook's first cell (the notebook actually uses these
`tensorflow.keras` versions below, not a duplicate PyTorch copy).
"""

from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Permute,
    SeparableConv2D,
    SpatialDropout2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2


def _resolve_dropout(dropout_type):
    if dropout_type == "SpatialDropout2D":
        return SpatialDropout2D
    if dropout_type == "Dropout":
        return Dropout
    raise ValueError("dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.")


def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64,
           F1=8, D=2, F2=16, norm_rate=0.25, dropoutType="Dropout"):
    """Lawhern et al. EEGNet (compact CNN for EEG decoding)."""
    dropout_layer = _resolve_dropout(dropoutType)

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding="same", input_shape=(Chans, Samples, 1), use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.0))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropout_layer(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropout_layer(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)
    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)


def EEGNet_SSVEP(nb_classes=12, Chans=8, Samples=256, dropoutRate=0.5, kernLength=256,
                  F1=96, D=1, F2=96, dropoutType="Dropout"):
    """EEGNet variant tuned for SSVEP classification (wider kernels/filters)."""
    dropout_layer = _resolve_dropout(dropoutType)

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding="same", input_shape=(Chans, Samples, 1), use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.0))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropout_layer(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropout_layer(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)
    dense = Dense(nb_classes, name="dense")(flatten)
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)


def EEGNet_legacy(nb_classes, Chans=64, Samples=128, regRate=0.0001,
                   dropoutRate=0.25, kernels=((2, 32), (8, 4)), strides=(2, 4)):
    """Earlier EEGNet formulation, kept for comparison (was `EEGNet_old`)."""
    input_main = Input((Chans, Samples))
    layer1 = Conv2D(16, (Chans, 1), input_shape=(Chans, Samples, 1),
                     kernel_regularizer=l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation("elu")(layer1)
    layer1 = Dropout(dropoutRate)(layer1)

    permute1 = Permute((2, 1, 3))(layer1)

    layer2 = Conv2D(4, kernels[0], padding="same", kernel_regularizer=l1_l2(l1=0.0, l2=regRate), strides=strides)(permute1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation("elu")(layer2)
    layer2 = Dropout(dropoutRate)(layer2)

    layer3 = Conv2D(4, kernels[1], padding="same", kernel_regularizer=l1_l2(l1=0.0, l2=regRate), strides=strides)(layer2)
    layer3 = BatchNormalization()(layer3)
    layer3 = Activation("elu")(layer3)
    layer3 = Dropout(dropoutRate)(layer3)

    flatten = Flatten(name="flatten")(layer3)
    dense = Dense(nb_classes, name="dense")(flatten)
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet(nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
    """Schirrmeister et al. DeepConvNet."""
    input_main = Input((Chans, Samples, 1))

    block1 = Conv2D(25, (1, 5), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation("elu")(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2 = Activation("elu")(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3 = Activation("elu")(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4 = Activation("elu")(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation("softmax")(dense)

    return Model(inputs=input_main, outputs=softmax)


def _square(x):
    return K.square(x)


def _log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    """Schirrmeister et al. ShallowConvNet."""
    input_main = Input((Chans, Samples, 1))

    block1 = Conv2D(40, (1, 13), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False, kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(_square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(_log)(block1)
    block1 = Dropout(dropoutRate)(block1)

    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation("softmax")(dense)

    return Model(inputs=input_main, outputs=softmax)
