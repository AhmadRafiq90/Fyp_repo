import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# ---------------------------
# CBAM Attention Module
# ---------------------------
def cbam_module(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    
    # Channel Attention
    shared_dense_one = layers.Dense(channel // reduction_ratio, activation='relu',
                                    kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_dense_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)
    
    channel_attention = layers.Add()([avg_pool, max_pool])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    x = layers.Multiply()([input_tensor, channel_attention])
    
    # Spatial Attention
    avg_pool_spatial = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool_spatial = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = layers.Conv2D(filters=1, kernel_size=7, padding='same', 
                                      activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    x = layers.Multiply()([x, spatial_attention])
    return x

# ---------------------------
# Custom Model Architecture
# ---------------------------
def build_custom_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Rescaling(1./255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = cbam_module(x)
    
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = cbam_module(x)
    
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = cbam_module(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# ---------------------------
# Data Preparation
# ---------------------------
data_dir = '/mnt/d/potential datasets/archive_2/plantvillage dataset/color/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 6
EPOCHS = 100

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation'
)

num_classes = len(train_generator.class_indices)

# ---------------------------
# Compute Class Weights
# ---------------------------
y_train = train_generator.labels
computed_class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(computed_class_weights))

# ---------------------------
# Build and Compile Model
# ---------------------------
tf.keras.backend.clear_session()
model = build_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.summary()

# ---------------------------
# Callbacks
# ---------------------------
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint("cbam_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ---------------------------
# Training
# ---------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ---------------------------
# Save Model
# ---------------------------
model.save("final_custom_leaf_model.h5")

# ---------------------------
# Plot Graphs
# ---------------------------
def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# Display Graphs
plot_metrics(history)

# ---------------------------
# Cleanup to Free Memory
# ---------------------------
tf.keras.backend.clear_session()
gc.collect()
