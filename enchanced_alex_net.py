import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os
import datetime

def load_data(train_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    ImageGenerator for Data Augmentation and Rescaling
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,  # 20% for validation
        rotation_range=40,                  # Rotate images by up to 40 degrees
        width_shift_range=0.2,              # Shift images horizontally by 20%
        height_shift_range=0.2,             # Shift images vertically by 20%
        shear_range=0.2,                    # Apply shear transformations
        zoom_range=0.2,                     # Apply random zoom
        horizontal_flip=True,               # Randomly flip images horizontally
        fill_mode='nearest'                 # Fill in missing pixels
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'                   # Use this subset for training
    )
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'                 # Use this subset for validation
    )

    return train_generator, validation_generator

def build_alexnet_model(input_shape, num_classes):
    """
    AlexNet Model with Enhancements
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', kernel_regularizer=l2(1e-4), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),

        # Second convolutional block
        layers.Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),

        # Third convolutional block
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4))
    ])
    return model

def train_model(model, train_generator, validation_generator, epochs, checkpoint_dir):
    """
    Train the AlexNet Model
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "test_alex.keras"),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )

    # Compile the model
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint_callback, reduce_lr],
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )

    return history

def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_shape = (224, 224, 3)  # Standard AlexNet input size
    train_dir = "/mnt/e/potential datasets/archive_2/plantvillage dataset/color/"  # Update with your dataset path
    checkpoint_dir = f"./checkpoints/model_{current_time}"

    train_generator, validation_generator = load_data(train_dir)
    num_classes = len(train_generator.class_indices)

    # Build and train the AlexNet model
    model = build_alexnet_model(input_shape, num_classes)
    history = train_model(model, train_generator, validation_generator, epochs=100, checkpoint_dir=checkpoint_dir)

    # Print final training and validation accuracy
    print(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

    # Save the model
    model.save("test_alex.keras")

if __name__ == "__main__":
    main()
