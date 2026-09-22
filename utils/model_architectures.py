"""The five spatio-temporal architectures compared in the thesis.

ConvLSTM (section 5.5), U-Net + LSTM (5.6), Attention U-Net + ConvLSTM (5.7),
Swin Spatio-Temporal Transformer (5.8) and ViT/ViViT (5.9), together with the
``MODEL_REGISTRY`` / ``build_model`` dispatcher.  TensorFlow-dependent.
"""

from __future__ import annotations

from typing import Callable, Dict

from .tf_env import silence_tf

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from .model_losses import combined_loss, dice_coefficient, iou_metric
from .pipeline_utils import CLIPNORM, IMG_HEIGHT, IMG_WIDTH, LEARNING_RATE

silence_tf()


def _adam():
    return Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)


def build_convlstm(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """Architecture 1 – standalone ConvLSTM (thesis Figure 5.1)."""
    inputs = layers.Input(shape=(seq_len, *input_shape))

    x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, 3, padding="same", activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)

    x = layers.ConvLSTM2D(
        128,
        (3, 3),
        padding="same",
        return_sequences=True,
        dropout=0.3,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(
        64,
        (3, 3),
        padding="same",
        return_sequences=False,
        dropout=0.3,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name=f"ConvLSTM_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


def _conv_bn(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def build_unet_lstm(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """
    Architecture 2 – U-Net with a ConvLSTM bottleneck (thesis Figure 5.2).

    The encoder is applied per-frame (TimeDistributed); the bottleneck
    models temporal dependencies; the decoder restores resolution using
    skip connections.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))
    td = layers.TimeDistributed

    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(inputs)
    e1 = td(layers.BatchNormalization())(e1)
    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(e1)
    e1 = td(layers.BatchNormalization())(e1)
    p1 = td(layers.MaxPooling2D(2))(e1)
    p1 = td(layers.Dropout(0.15))(p1)

    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(p1)
    e2 = td(layers.BatchNormalization())(e2)
    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(e2)
    e2 = td(layers.BatchNormalization())(e2)
    p2 = td(layers.MaxPooling2D(2))(e2)
    p2 = td(layers.Dropout(0.2))(p2)

    e3 = td(layers.Conv2D(128, 3, padding="same", activation="relu"))(p2)
    e3 = td(layers.BatchNormalization())(e3)
    e3 = td(layers.Conv2D(128, 3, padding="same", activation="relu"))(e3)
    e3 = td(layers.BatchNormalization())(e3)
    p3 = td(layers.MaxPooling2D(2))(e3)
    p3 = td(layers.Dropout(0.2))(p3)

    b = td(layers.Conv2D(256, 3, padding="same", activation="relu"))(p3)
    b = td(layers.BatchNormalization())(b)
    b = layers.ConvLSTM2D(256, (3, 3), padding="same", return_sequences=True, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)
    b = layers.ConvLSTM2D(128, (3, 3), padding="same", return_sequences=False, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)

    take_last = layers.Lambda(lambda t: t[:, -1, ...], name="take_last_frame")

    d3 = layers.UpSampling2D(2)(b)
    d3 = layers.Concatenate()([d3, take_last(e3)])
    d3 = _conv_bn(d3, 128, 0.2)
    d3 = _conv_bn(d3, 128)

    d2 = layers.UpSampling2D(2)(d3)
    d2 = layers.Concatenate()([d2, take_last(e2)])
    d2 = _conv_bn(d2, 64, 0.2)
    d2 = _conv_bn(d2, 64)

    d1 = layers.UpSampling2D(2)(d2)
    d1 = layers.Concatenate()([d1, take_last(e1)])
    d1 = _conv_bn(d1, 32)

    d1 = layers.Conv2D(16, 3, padding="same", activation="relu")(d1)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d1)

    model = models.Model(inputs, outputs, name=f"UNetLSTM_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


class AttentionGate(layers.Layer):
    """
    Additive attention gate (Oktay et al., 2018) used on U-Net skip
    connections.  ``g`` is the gating (decoder) signal and ``s`` the skip
    feature map; the gate highlights the spatially relevant regions.
    """

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.Wg = layers.Conv2D(filters, 1, padding="same")
        self.Ws = layers.Conv2D(filters, 1, padding="same")
        self.psi = layers.Conv2D(1, 1, padding="same")
        self.relu = layers.Activation("relu")
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, inputs):
        g, s = inputs
        x = self.relu(self.Wg(g) + self.Ws(s))
        x = self.sigmoid(self.psi(x))
        return s * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters})
        return cfg


def build_attention_unet_convlstm(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """
    Architecture 3 – Attention U-Net with ConvLSTM bottleneck
    (thesis Figure 5.3).  The best model for yearly and quarterly horizons.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))
    td = layers.TimeDistributed

    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(inputs)
    e1 = td(layers.BatchNormalization())(e1)
    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(e1)
    e1 = td(layers.BatchNormalization())(e1)
    p1 = td(layers.MaxPooling2D(2))(e1)
    p1 = td(layers.Dropout(0.15))(p1)

    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(p1)
    e2 = td(layers.BatchNormalization())(e2)
    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(e2)
    e2 = td(layers.BatchNormalization())(e2)
    p2 = td(layers.MaxPooling2D(2))(e2)
    p2 = td(layers.Dropout(0.2))(p2)

    e3 = td(layers.Conv2D(128, 3, padding="same", activation="relu"))(p2)
    e3 = td(layers.BatchNormalization())(e3)
    p3 = td(layers.MaxPooling2D(2))(e3)
    p3 = td(layers.Dropout(0.2))(p3)

    b = td(layers.Conv2D(256, 3, padding="same", activation="relu"))(p3)
    b = td(layers.BatchNormalization())(b)
    b = layers.ConvLSTM2D(256, (3, 3), padding="same", return_sequences=True, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)
    b = layers.ConvLSTM2D(128, (3, 3), padding="same", return_sequences=False, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)

    take_last = layers.Lambda(lambda t: t[:, -1, ...])

    d3 = layers.UpSampling2D(2)(b)
    a3 = AttentionGate(64)([d3, take_last(e3)])
    d3 = layers.Concatenate()([d3, a3])
    d3 = _conv_bn(d3, 128, 0.2)

    d2 = layers.UpSampling2D(2)(d3)
    a2 = AttentionGate(32)([d2, take_last(e2)])
    d2 = layers.Concatenate()([d2, a2])
    d2 = _conv_bn(d2, 64, 0.2)

    d1 = layers.UpSampling2D(2)(d2)
    a1 = AttentionGate(16)([d1, take_last(e1)])
    d1 = layers.Concatenate()([d1, a1])
    d1 = _conv_bn(d1, 32)

    d1 = layers.Conv2D(16, 3, padding="same", activation="relu")(d1)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d1)

    model = models.Model(inputs, outputs, name=f"AttentionUNetConvLSTM_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


def build_swin_st(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """
    Architecture 4 – Swin Spatio-Temporal Transformer (thesis Figure 5.4).

    A lightweight Swin-style hierarchical encoder (window attention +
    patch merging) shared across frames, a temporal fusion head, and a
    U-Net-style convolutional decoder.  Implemented with Keras layers so
    it runs on the same TensorFlow stack as the other models.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))

    def patch_embed(x):
        return layers.Conv2D(64, 4, strides=4, padding="same")(x)

    def window_attention(x, dim, heads):
        """Simplified shifted-window attention: local 3×3 conv mixing
        followed by global channel attention (efficient proxy for Swin)."""
        y = layers.Conv2D(dim, 3, padding="same", activation="gelu")(x)
        h, w = y.shape[1], y.shape[2]
        y_3d = layers.Reshape((h * w, dim))(y)
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=0.1)(
            y_3d, y_3d
        )
        attn = layers.Reshape((h, w, dim))(attn)
        return layers.Add()([x, attn])

    frames = [layers.Lambda(lambda t, i=i: t[:, i, ...])(inputs) for i in range(seq_len)]

    encoded = []
    for f in frames:
        x = patch_embed(f)  # (64, 64, 64)
        x = window_attention(x, 64, 2)  # stage 1
        skip1 = x
        x = layers.MaxPooling2D(2)(x)  # (32, 32, 64)
        x = layers.Conv2D(128, 3, padding="same", activation="gelu")(x)
        x = window_attention(x, 128, 4)
        skip2 = x
        x = layers.MaxPooling2D(2)(x)  # (16, 16, 128)
        x = layers.Conv2D(256, 3, padding="same", activation="gelu")(x)
        x = window_attention(x, 256, 8)
        encoded.append((x, skip1, skip2))

    h_f, w_f = encoded[-1][0].shape[1], encoded[-1][0].shape[2]
    frame_vecs = [layers.Reshape((1, 256))(layers.GlobalAveragePooling2D()(e[0])) for e in encoded]
    stacked = layers.Concatenate(axis=1)(frame_vecs)  # (batch, T, 256)
    t = layers.Conv1D(256, 3, padding="same", activation="gelu")(stacked)
    t = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(t, t)
    t = layers.GlobalAveragePooling1D()(t)  # aggregate over time -> (batch, 256)
    t = layers.Dense(h_f * w_f * 256)(t)
    t = layers.Reshape((h_f, w_f, 256))(t)

    d = layers.UpSampling2D(2)(t)  # 32
    d = layers.Concatenate()([d, encoded[-1][2]])
    d = _conv_bn(d, 128)

    d = layers.UpSampling2D(2)(d)  # 64
    d = layers.Concatenate()([d, encoded[-1][1]])
    d = _conv_bn(d, 64)

    d = layers.UpSampling2D(2)(d)  # 128
    d = _conv_bn(d, 32)
    d = layers.UpSampling2D(2)(d)  # 256
    d = _conv_bn(d, 16)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d)
    model = models.Model(inputs, outputs, name=f"SwinST_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


def build_vit_st(
    seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), patch_size: int = 16, embed_dim: int = 128
):
    """
    Architecture 5 – ViT / ViViT spatio-temporal model (thesis Figure 5.5).

    Per-frame patch tokenisation → spatial transformer → global average
    pooling → temporal transformer → dense projection → convolutional
    decoder.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))
    h_out = input_shape[0] // patch_size
    w_out = input_shape[1] // patch_size
    n_patches = h_out * w_out

    patch_embed = layers.Conv2D(
        embed_dim, patch_size, strides=patch_size, padding="valid", name="patch_embed"
    )
    token_norm = layers.LayerNormalization(name="token_norm")
    spatial_mha = layers.MultiHeadAttention(
        num_heads=4, key_dim=embed_dim // 4, dropout=0.1, name="spatial_mha"
    )
    post_norm = layers.LayerNormalization(name="post_norm")
    token_pool = layers.GlobalAveragePooling1D(name="token_pool")

    frames = [layers.Lambda(lambda t, i=i: t[:, i, ...])(inputs) for i in range(seq_len)]
    frame_tokens_list = []
    for f in frames:
        x = patch_embed(f)
        x = layers.Reshape((n_patches, embed_dim))(x)
        x = token_norm(x)
        x = spatial_mha(x, x)
        x = post_norm(x)
        x = token_pool(x)  # (B, embed_dim)
        frame_tokens_list.append(layers.Reshape((1, embed_dim))(x))

    frame_tokens = layers.Concatenate(axis=1)(frame_tokens_list)

    t = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim // 4, dropout=0.1)(
        frame_tokens, frame_tokens
    )
    t = layers.LayerNormalization()(t)
    t = layers.Dense(256, activation="gelu")(t)
    t = layers.GlobalAveragePooling1D()(t)  # (B, 256)
    t = layers.Dense(h_out * w_out * embed_dim, activation="gelu")(t)
    t = layers.Reshape((h_out, w_out, embed_dim))(t)

    d = layers.Conv2DTranspose(128, 2, strides=2, padding="same", activation="relu")(t)  # 32
    d = layers.Conv2DTranspose(64, 2, strides=2, padding="same", activation="relu")(d)  # 64
    d = layers.Conv2DTranspose(32, 2, strides=2, padding="same", activation="relu")(d)  # 128
    d = layers.Conv2DTranspose(16, 2, strides=2, padding="same", activation="relu")(d)  # 256
    d = _conv_bn(d, 16)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d)
    model = models.Model(inputs, outputs, name=f"ViT_ST_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


MODEL_REGISTRY: Dict[str, Callable[[int], "tf.keras.Model"]] = {
    "convlstm": build_convlstm,
    "unet_lstm": build_unet_lstm,
    "attention_unet_convlstm": build_attention_unet_convlstm,
    "swin_st": build_swin_st,
    "vit_st": build_vit_st,
}


def build_model(name: str, seq_len: int):
    """Build and compile a model from the registry by its string name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. " f"Options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seq_len)
