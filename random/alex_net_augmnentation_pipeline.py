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

def create_data_pipeline(train_dir, img_size=(256, 256), batch_size=8, validation_split=0.2):
    """Data pipeline with advanced augmentation and preprocessing"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        channel_shift_range=50.0
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def build_generalizable_model(input_shape, num_classes):
    """Enhanced AlexNet-style model with generalization modifications"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),
        layers.SpatialDropout2D(0.25),
        
        layers.Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),
        layers.SpatialDropout2D(0.35),
        
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', depthwise_regularizer=l2(1e-4)),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', depthwise_regularizer=l2(1e-4)),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', depthwise_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        layers.Dense(1024, activation='swish', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.6),
        layers.Dense(1024, activation='swish', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.6),
        
        layers.Dense(num_classes, activation='softmax', activity_regularizer=l2(1e-5))
    ])
    return model

def enhanced_training_pipeline(model, train_gen, val_gen, checkpoint_dir):
    """Training pipeline with class balancing and callbacks"""
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "alexnet_augmentationpipe_model.keras"),
                        monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7)
    ]
    
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4, global_clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')])
    
    history = model.fit(
        train_gen,
        epochs=150,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        verbose=2
    )
    return history

def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_dir = "/mnt/d/potential datasets/archive_2/plantvillage dataset/color/"
    checkpoint_dir = f"./checkpoints/plant_model_{current_time}"
    img_size = (256, 256)
    
    train_gen, val_gen = create_data_pipeline(train_dir, img_size=img_size)
    model = build_generalizable_model(input_shape=img_size + (3,), num_classes=len(train_gen.class_indices))
    
    history = enhanced_training_pipeline(model, train_gen, val_gen, checkpoint_dir)
    
    print(f"\nBest Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")
    print(f"Final Top-3 Validation Accuracy: {history.history['val_top3_accuracy'][-1] * 100:.2f}%")
    
    model.save("generalizable_plant_model.keras")

if __name__ == "__main__":
    main()
