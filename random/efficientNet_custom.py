import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Add, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Directories
train_dir = "/mnt/d/potential datasets/archive_2/plantvillage dataset/color/"
val_dir = "/mnt/d/potential datasets/archive_2/plantvillage dataset/test"
img_size = (224, 224)
batch_size = 8

# Data Augmentation and Preprocessing
def add_gaussian_noise(img):
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.05, dtype=tf.float32)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)  # Ensure values remain in valid range
    return img

data_gen = ImageDataGenerator(
    preprocessing_function=add_gaussian_noise,
    rescale=1./255,  # Ensure images are in [0,1] range
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

data_gen_val = ImageDataGenerator(rescale=1./255)  # Ensure validation images are also in [0,1]

train_gen = data_gen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='sparse'
)
val_gen = data_gen_val.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='sparse'
)

num_classes = len(train_gen.class_indices)

# Compute Class Weights
train_classes = np.array(train_gen.classes)
class_weights = compute_class_weight("balanced", classes=np.unique(train_classes), y=train_classes)
class_weights = dict(enumerate(class_weights))

# Custom EfficientNet-like Model
def mb_conv_block(x, filters, expansion_factor, stride):
    in_channels = x.shape[-1]
    expanded = Conv2D(in_channels * expansion_factor, (1, 1), padding='same', use_bias=False)(x)
    expanded = BatchNormalization()(expanded)
    expanded = tf.nn.swish(expanded)

    depthwise = DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(expanded)
    depthwise = BatchNormalization()(depthwise)
    depthwise = tf.nn.swish(depthwise)

    squeezed = Conv2D(filters, (1, 1), padding='same', use_bias=False)(depthwise)
    squeezed = BatchNormalization()(squeezed)

    if stride == 1 and in_channels == filters:
        x = Add()([x, squeezed])
    return squeezed

# Build Model
input_layer = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False)(input_layer)
x = BatchNormalization()(x)
x = tf.nn.swish(x)

x = mb_conv_block(x, filters=16, expansion_factor=1, stride=1)
x = mb_conv_block(x, filters=24, expansion_factor=6, stride=2)
x = mb_conv_block(x, filters=40, expansion_factor=6, stride=2)
x = mb_conv_block(x, filters=80, expansion_factor=6, stride=2)
x = mb_conv_block(x, filters=112, expansion_factor=6, stride=1)
x = mb_conv_block(x, filters=192, expansion_factor=6, stride=2)
x = mb_conv_block(x, filters=320, expansion_factor=6, stride=1)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    class_weight=class_weights,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[reduce_lr, early_stop]
)

# Save Model
model.save("custom_efficientnet_plant", save_format="tf")
