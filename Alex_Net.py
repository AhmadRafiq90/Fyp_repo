import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import datetime

def load_data(train_dir, img_size=(224, 224), batch_size=16, validation_split=0.2):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator


def build_alexnet_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', kernel_initializer='he_normal', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.6),
        layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.6),
        layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')
    ])
    return model


def train_model(model, train_generator, validation_generator, epochs, patience, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_filepath = os.path.join(checkpoint_dir, "model.keras")
    
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        verbose=1,
        min_lr=1e-6
    )
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint_callback, lr_scheduler]
    )
    
    return history


def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_shape = (224, 224, 3)
    train_dir = "/mnt/e/potential datasets/archive_2/plantvillage dataset/color/"
    checkpoint_dir = f"./checkpoints/model_{current_time}"
    
    train_generator, validation_generator = load_data(train_dir)
    num_classes = len(train_generator.class_indices)
    model = build_alexnet_model(input_shape, num_classes)
    
    history = train_model(model, train_generator, validation_generator, epochs=100, patience=20, checkpoint_dir=checkpoint_dir)
    
    print(f"Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    
    model.save("Species_detector.keras")


if __name__ == "__main__":
    main()
