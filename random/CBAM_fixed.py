import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

# ---------------------------
# CBAM Attention Module (Fixed)
# ---------------------------
def cbam_module(input_tensor, reduction_ratio=8):
    channel = input_tensor.shape[-1]
    
    # Channel Attention Module
    shared_dense_one = layers.Dense(channel // reduction_ratio, 
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_dense_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    
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
    
    # Spatial Attention Module: wrap tf.reduce_mean and tf.reduce_max in Lambda layers
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
    
    # Data Augmentation (also applied in generator) and rescaling
    x = layers.Rescaling(1./255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    
    # Block 1
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = cbam_module(x)
    
    # Block 2
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu', 
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = cbam_module(x)
    
    # Block 3
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu', 
                      kernel_regularizer=regularizers.l2(0.001))(x)
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
# MixUp Augmentation Functions
# ---------------------------
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma1 = tf.random.gamma(shape=[size], alpha=concentration_0)
    gamma2 = tf.random.gamma(shape=[size], alpha=concentration_1)
    return gamma1 / (gamma1 + gamma2)

def mixup(batch_x, batch_y, alpha=0.2):
    batch_size = tf.shape(batch_x)[0]
    lam = sample_beta_distribution(batch_size, alpha, alpha)
    lam_x = tf.reshape(lam, (batch_size, 1, 1, 1))  # For images
    lam_y = tf.reshape(lam, (batch_size, 1))          # For labels
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_x = lam_x * batch_x + (1 - lam_x) * tf.gather(batch_x, index)
    mixed_y = lam_y * batch_y + (1 - lam_y) * tf.gather(batch_y, index)
    return mixed_x, mixed_y

def mixup_generator(generator, alpha=0.2):
    while True:
        batch_x, batch_y = next(generator)
        yield mixup(batch_x, batch_y, alpha)

# ---------------------------
# Data Preparation
# ---------------------------
# Ensure your dataset folder ('./color/') is structured with each class in its own subfolder.
data_dir = './color/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch size helps when data is limited
EPOCHS = 100

# Heavy augmentation using ImageDataGenerator
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
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)

# ---------------------------
# Build and Compile Model
# ---------------------------
model = build_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)

# Use AdamW optimizer with weight decay and label smoothing in the loss function
optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()

# ---------------------------
# Callbacks for Adaptive Training
# ---------------------------
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ---------------------------
# Training with MixUp Augmentation
# ---------------------------
train_mixup = mixup_generator(train_generator, alpha=0.2)

history = model.fit(
    train_mixup,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ---------------------------
# Save Final Model and Print Metrics
# ---------------------------
model.save("final_custom_leaf_model.h5")
print("Training complete. Model saved as final_custom_leaf_model.h5")

final_train_acc = history.history['accuracy'][-1] * 100 if 'accuracy' in history.history else None
final_val_acc = history.history['val_accuracy'][-1] * 100 if 'val_accuracy' in history.history else None
if final_train_acc is not None:
    print(f"Final Training Accuracy: {final_train_acc:.2f}%")
if final_val_acc is not None:
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")

# Plot training history for review
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.tight_layout()
plt.show()
