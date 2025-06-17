import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import datetime
from sklearn.utils import class_weight

def create_advanced_augmentation(img_size=(300, 300)):
    """Enhanced augmentation pipeline for small datasets"""
    return ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        channel_shift_range=50.0,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        preprocessing_function=lambda x: x + tf.random.normal(x.shape, 0, 0.1)  # Add mild noise
    )

def mb_conv_block(x, filters, kernel_size, strides=1, expansion=4, se_ratio=0.25):
    """Custom MBConv block with simplified squeeze-excitation"""
    # Expansion phase
    expanded_filters = filters * expansion
    x = layers.Conv2D(expanded_filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    # Squeeze-excitation
    se_filters = max(1, int(filters * se_ratio))
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(se_filters, activation='swish')(se)
    se = layers.Dense(expanded_filters, activation='sigmoid')(se)
    x = layers.multiply([x, se])

    # Projection
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    return x

def build_custom_efficientnet(input_shape, num_classes):
    """Optimized EfficientNet-style architecture for plant classification"""
    inputs = layers.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)

    # MBConv blocks
    x = mb_conv_block(x, 16, 3, expansion=1)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = mb_conv_block(x, 24, 3)
    x = mb_conv_block(x, 24, 3)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = mb_conv_block(x, 40, 5)
    x = mb_conv_block(x, 40, 5)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Head
    x = layers.Conv2D(1280, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier with regularization
    x = layers.Dense(512, activation='swish', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='swish', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def train_model(model, train_dir, img_size=(300, 300), epochs=200):
    """Complete training pipeline with generalization safeguards"""
    # Data generators
    train_datagen = create_advanced_augmentation(img_size)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=16,  # Smaller batches for better regularization
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Class balancing
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True),
        ModelCheckpoint('efficient_model.keras', monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-6)
    ]
    
    # Optimizer with weight decay
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
    
    # Compile with label smoothing
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy', 
                        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')])
    
    # Training
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        verbose=2)
    return history

def main():
    # Configuration
    train_dir = "/mnt/e/potential datasets/archive_2/plantvillage dataset/color/"
    img_size = (300, 300)  # Larger input size for better feature extraction
    
    # Build model
    model = build_custom_efficientnet(input_shape=img_size + (3,), 
                                     num_classes=len(ImageDataGenerator().flow_from_directory(
                                         train_dir, 
                                         class_mode='categorical').class_indices))
    
    # Train
    history = train_model(model, train_dir, img_size)
    
    # Results
    print(f"\nBest Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")
    print(f"Top-3 Validation Accuracy: {history.history['val_top3_acc'][-1] * 100:.2f}%")
    
    # Save final model
    model.save("custom_efficientnet_plant.h5")

if __name__ == "__main__":
    main()