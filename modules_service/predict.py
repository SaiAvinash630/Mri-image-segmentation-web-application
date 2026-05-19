"""
Model Prediction Module (KERAS ONLY VERSION)
"""

import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# =========================================================
# 🔹 IMPORTS
# =========================================================
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# =========================================================
# 🔹 LOSSES & METRICS
# =========================================================
sobel_x = tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=tf.float32)
sobel_x = tf.reshape(sobel_x, [3,3,1,1])

sobel_y = tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=tf.float32)
sobel_y = tf.reshape(sobel_y, [3,3,1,1])

bce = tf.keras.losses.BinaryCrossentropy()


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)

    return tf.reduce_mean(
        (2. * intersection + smooth) /
        (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def boundary_loss(y_true, y_pred):
    gx_true = tf.nn.conv2d(y_true, sobel_x, [1,1,1,1], "SAME")
    gy_true = tf.nn.conv2d(y_true, sobel_y, [1,1,1,1], "SAME")

    gx_pred = tf.nn.conv2d(y_pred, sobel_x, [1,1,1,1], "SAME")
    gy_pred = tf.nn.conv2d(y_pred, sobel_y, [1,1,1,1], "SAME")

    grad_true = tf.sqrt(gx_true**2 + gy_true**2 + 1e-6)
    grad_pred = tf.sqrt(gx_pred**2 + gy_pred**2 + 1e-6)

    return tf.reduce_mean(tf.abs(grad_true - grad_pred))


def combined_loss(y_true, y_pred):
    return (
        0.3 * bce(y_true, y_pred)
        + 0.5 * dice_loss(y_true, y_pred)
        + 0.2 * boundary_loss(y_true, y_pred)
    )

# =========================================================
# 🔹 CUSTOM LAYER
# =========================================================
class ModalityWeighting(layers.Layer):

    def __init__(self, **kwargs):
        super(ModalityWeighting, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(1,1,1,input_shape[-1]),
            initializer=tf.keras.initializers.Constant([0.4,0.3,0.3]),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

    def call(self, x):
        return x * self.w

    def get_config(self):
        config = super().get_config()
        return config
from tensorflow.keras import layers, models
import tensorflow as tf

# =========================================================
# 🔹 Residual and DSE Blocks
# =========================================================


def norm_layer(x):
    return layers.GroupNormalization(groups=8, axis=-1)(x)

def residual_block(x, filters):
    shortcut = layers.Conv2D(filters, (1,1), padding='same')(x)
    shortcut = norm_layer(shortcut)   # ✅ GroupNorm

    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = norm_layer(x)                 # ✅ GroupNorm
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = norm_layer(x)                 # ✅ GroupNorm

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
def channel_squeeze_excitation(x, reduction=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // reduction, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = layers.Multiply()([x, se])
    return x

# --------------------------
# Spatial Squeeze & Excitation (sSE)
# --------------------------
def spatial_squeeze_excitation(x):
    se = layers.Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)
    x = layers.Multiply()([x, se])
    return x

# --------------------------
# Dual Squeeze & Excitation (DSE / scSE)
# --------------------------
def dual_squeeze_excitation(x, reduction=16):
    cse = channel_squeeze_excitation(x, reduction)
    sse = spatial_squeeze_excitation(x)
    x = layers.Add()([cse, sse])
    return x

def ASPP(x, filters=256):

    y1 = layers.Conv2D(filters, 1, padding='same')(x)
    y1 = norm_layer(y1)
    y1 = layers.Activation('relu')(y1)

    y2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6)(x)
    y2 = norm_layer(y2)
    y2 = layers.Activation('relu')(y2)

    y3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12)(x)
    y3 = norm_layer(y3)
    y3 = layers.Activation('relu')(y3)

    y4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18)(x)
    y4 = norm_layer(y4)
    y4 = layers.Activation('relu')(y4)

    y = layers.concatenate([y1, y2, y3, y4])

    y = layers.Conv2D(filters, 1, padding='same')(y)
    y = norm_layer(y)
    y = layers.Activation('relu')(y)

    return y
# =========================================================
# 🔹 ResUNet + DSE Model
# =========================================================

def ResUNet_DSE(input_shape=(128,128,3)):
    inputs = layers.Input(input_shape)
    x_in = ModalityWeighting()(inputs)

    # Encoder
    c1 = residual_block(x_in, 32)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = residual_block(p2, 128)
    p3 = layers.MaxPooling2D((2,2))(c3)

    c4 = residual_block(p3, 256)
    p4 = layers.MaxPooling2D((2,2))(c4)

    # Bottleneck with DSE
    c5 = residual_block(p4, 512)

    c5 = ASPP(c5)  # 🔹 multi-scale context

    c5 = layers.Dropout(0.2)(c5)

    c5 = dual_squeeze_excitation(c5)  # 🔹 attention refinement

    # Decoder
    u6 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = residual_block(u6, 256)

    u7 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = residual_block(u7, 128)

    u8 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = residual_block(u8, 64)

    u9 = layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = residual_block(u9, 32)

    # Output layer
    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = models.Model(inputs, outputs, name="ResUNet_DSE")
    return model
from tensorflow.keras.optimizers import Adam

def precision_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)

    return tp / (tp + fp + 1e-6)


def recall_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    return tp / (tp + fn + 1e-6)


model = ResUNet_DSE(input_shape=(128,128,3))
# =========================================================
# 🔹 MODEL CLASS
# =========================================================
class StrokeDetectionModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

        try:
            self.load_model(model_path)   # 🔥 USE CORRECT FUNCTION

        except Exception as e:
            print("❌ Model loading failed:", str(e))
            self.is_loaded = False

    # -----------------------------
    # ✅ LOAD MODEL (KERAS)
    # -----------------------------
    def load_model(self, model_path: str):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        custom_objects = {
            "ModalityWeighting": ModalityWeighting,
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient
        }

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )

        self.model.compile(
            optimizer=Adam(1e-4),
            loss=combined_loss,
            metrics=[dice_coefficient]
        )

        self.is_loaded = True
        print("✅ Model loaded successfully (REAL)")

    # -----------------------------
    # ✅ PREPROCESSED PREDICTION
    # -----------------------------
    def predict_preprocessed(
        self,
        preprocessed_data: np.ndarray,
        output_path: Optional[str] = None
    ) -> np.ndarray:

        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        try:
            logger.info(f"🧠 Input shape: {preprocessed_data.shape}, dtype: {preprocessed_data.dtype}")
            logger.info(f"🧠 Data range: min={preprocessed_data.min():.4f}, max={preprocessed_data.max():.4f}")

            if len(preprocessed_data.shape) != 4:
                raise ValueError(f"Expected shape (S,128,128,3), got {preprocessed_data.shape}")

            # ✅ Cast to float32 first
            preprocessed_data = preprocessed_data.astype(np.float32)

            # ✅ Only normalize if data is NOT already in [0,1]
            data_max = np.max(preprocessed_data)
            if data_max > 1.0:
                logger.info(f"Normalizing data (max was {data_max:.4f})")
                preprocessed_data = preprocessed_data / data_max
            else:
                logger.info("Data already in [0,1], skipping normalization")

            # 🔥 Predict
            preds = self.model.predict(preprocessed_data, verbose=1)

            logger.info(f"✅ Prediction range: min={preds.min():.4f}, max={preds.max():.4f}")
            logger.info(f"✅ % pixels > 0.5: {(preds > 0.5).mean()*100:.2f}%")

            if preds.max() < 0.1:
                logger.warning("⚠️ All predictions near zero — possible normalization mismatch!")

            # ==========================================
            # FALSE POSITIVE SMALL DOT REMOVAL
            # ==========================================
            MIN_LESION_PIXELS = 30   # adjust this

            filtered_preds = preds.copy()

            for i in range(filtered_preds.shape[0]):

                binary_mask = filtered_preds[i, :, :, 0] > 0.5
                lesion_pixels = np.sum(binary_mask)

                if lesion_pixels < MIN_LESION_PIXELS:
                    logger.info(
                        f"Removing false positive in slice {i} "
                        f"(pixels={lesion_pixels})"
                    )
                    filtered_preds[i, :, :, 0] = 0.0

            if output_path:
                np.save(output_path, filtered_preds.astype(np.float32))
                logger.info(f"💾 Saved filtered mask: {output_path}")

            return filtered_preds
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise

