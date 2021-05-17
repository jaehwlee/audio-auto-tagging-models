# coding: utf-8
import numpy as np
import tensorflow as tf
from modules import HarmonicSTFT
from modules import se_fn, se_block, basic_block, rese_block
from modules import MusicSinc1D

class HarmonicCNN(tf.keras.Model):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=512,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw=None,
                dataset='mtat'):
        super(HarmonicCNN, self).__init__()

        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.hstft_bn = tf.keras.layers.BatchNormalization()

        # 2D CNN
        if dataset == 'mtat':
            from modules import ResNet_mtat as ResNet
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def call(self, x, training=False):
        # harmonic stft
        x = self.hstft_bn(self.hstft(x, training=training), training=training)

        # 2D CNN
        logits = self.conv_2d(x, training=training)

        return logits



def resemul(
    ms=False, block_type="se", amplifying_ratio=16, drop_rate=0.5, weight_decay=1e-4, num_classes=50,
):

    # Input&Reshape
    if block_type == 'se':
        block = se_block
    elif block_type == 'rese':
        block = rese_block
    elif block_type == 'res':
        block = rese_block
        amplifying_ratio = -1
    elif block_type == 'basic':
        block = basic_block
    inp = tf.keras.layers.Input(shape=(59049, 1))
    x = inp
    # Strided Conv
    num_features = 128
    if ms:
        filter_size = 2501
        filter_num = 256
        x = tf.keras.layers.ZeroPadding1D(padding=(filter_size-1)//2)(x)
        x = MusicSinc1D(filter_num, filter_size, 22050)(x)
    
    x = tf.keras.layers.Conv1D(
        num_features,
        kernel_size=3,
        strides=3,
        padding="valid",
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # rese-block
    layer_outputs = []
    for i in range(9):
        num_features *= 2 if (i == 2 or i == 8) else 1
        x = block(x, num_features, weight_decay, amplifying_ratio)
        layer_outputs.append(x)

    x = tf.keras.layers.Concatenate()([tf.keras.layers.GlobalMaxPool1D()(output) for output in layer_outputs[-3:]])
    x = tf.keras.layers.Dense(x.shape[-1], kernel_initializer="glorot_uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation("relu")(x)
    if drop_rate > 0.0:
        x = Dropout(drop_rate)(x)
    out = Dense(num_classes, activation="sigmoid", kernel_initializer="glorot_uniform")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model
