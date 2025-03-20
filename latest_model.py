import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import os
import datetime
import numpy as np
from sklearn.utils import class_weight

# Custom Focal Loss implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        loss = alpha * tf.keras.backend.pow(1 - y_pred, gamma) * cross_entropy
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
    return focal_loss_fn

def load_data(train_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """Enhanced ImageGenerator with advanced augmentation"""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        channel_shift_range=50.0,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def build_enhanced_alexnet(input_shape, num_classes):
    """Enhanced AlexNet with improved regularization"""
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', 
                     kernel_regularizer=l2(1e-4), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.2),

        # Conv Block 2
        layers.Conv2D(256, (5, 5), padding='same', activation='relu',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.3),

        # Conv Block 3
        layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(1e-4)),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(1e-4)),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.4),

        # Dense Layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.6),
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.6),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, train_generator, validation_generator, epochs, checkpoint_dir, class_weights):
    """Enhanced training routine"""
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "latest_model_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )

    # Use lower initial learning rate with AdamW
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
    
    # Compile with focal loss
    model.compile(optimizer=optimizer,
                loss=focal_loss(),
                metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint_callback, reduce_lr, early_stop],
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator),
        verbose=2
    )
    return history

def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_shape = (224, 224, 3)
    train_dir = "/mnt/e/potential datasets/archive_2/plantvillage dataset/color/"
    checkpoint_dir = f"./checkpoints/model_{current_time}"

    train_generator, validation_generator = load_data(train_dir)
    num_classes = len(train_generator.class_indices)

    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))

    # Build and train model
    model = build_enhanced_alexnet(input_shape, num_classes)
    history = train_model(model, train_generator, validation_generator,
                         epochs=150, checkpoint_dir=checkpoint_dir,
                         class_weights=class_weights)

    # Save final model
    model.save("enhanced_alexnet_final.keras")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")

if __name__ == "__main__":
    main()