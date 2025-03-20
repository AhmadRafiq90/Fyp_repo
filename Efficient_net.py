import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import datetime

# Load data with augmentation for small dataset
def load_data(train_dir, img_size=(224, 224), batch_size=16, validation_split=0.2):
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

# Build EfficientNetB0 model with transfer learning
def build_efficientnet_model(input_shape, num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',                # Use pretrained weights from ImageNet
        include_top=False,                 # Exclude the fully connected layers
        input_shape=input_shape            # Specify input shape (224, 224, 3)
    )
    
    # Freeze base model layers (do not train them)
    base_model.trainable = False

    model = models.Sequential([
        base_model,                        # Add the base EfficientNetB0 model
        layers.GlobalAveragePooling2D(),   # Pooling to flatten output
        layers.Dense(1024, activation='relu'),  # Fully connected layer with ReLU activation
        layers.Dropout(0.5),               # Dropout for regularization
        layers.Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])
    
    return model

# Train the model
def train_model(model, train_generator, validation_generator, epochs, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # ModelCheckpoint saves the best model based on validation accuracy
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.keras"),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # EarlyStopping to stop training if validation accuracy doesn't improve
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )

    return history

# Main function to initiate training
def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_shape = (224, 224, 3)  # EfficientNet input size
    train_dir = "/media/ahmad/New Volume/potential datasets/archive_2/plantvillage dataset/color/"  # Update with your dataset path
    checkpoint_dir = f"./checkpoints/efficientnetmodel_{current_time}"

    train_generator, validation_generator = load_data(train_dir)
    num_classes = len(train_generator.class_indices)

    # Build and train the EfficientNet model
    model = build_efficientnet_model(input_shape, num_classes)
    history = train_model(model, train_generator, validation_generator, epochs=100, checkpoint_dir=checkpoint_dir)

    # Print final training and validation accuracy
    print(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

    # Save the model
    model.save("final_efficientnet_model.keras")

if __name__ == "__main__":
    main()
