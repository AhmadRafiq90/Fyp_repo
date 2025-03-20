import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
import datetime

# Ensure TensorFlow uses GPU if available
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Image Parameters
IMAGE_SIZE = (227, 227)  # AlexNet input size
BATCH_SIZE = 7
EPOCHS = 100
NUM_CLASSES = 39  # Adjust based on your dataset

# Data Directory (Assuming same folder as script)
data_dir = "/mnt/d/potential datasets/archive_2/plantvillage dataset/color/"

# Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

# Load Training & Validation Data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)
# Define Custom CNN Model (Inspired by AlexNet)
def create_model():
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, activation="relu", input_shape=(227, 227, 3)),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(256, (5, 5), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(384, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),

        Conv2D(384, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),

        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),

        Dense(4096, activation="relu"),
        Dropout(0.5),

        Dense(NUM_CLASSES, activation="softmax")  # Output layer
    ])
    
    # Compile Model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model

# Create Model
model = create_model()
model.summary()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = f"./checkpoints/alexnet_model_{current_time}"
# Callbacks for Optimization
lr_scheduler = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
checkpoint_system = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "alexnet_augmentation_model.keras"),
                     monitor='val_accuracy', save_best_only=True, verbose=1)
# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stopping, checkpoint_system],
    verbose=1
)

# Save Model
model.save("custom_cnn_model.keras")
print("Training Complete and Model Saved!")

# Print Final Accuracy
final_train_acc = history.history["accuracy"][-1] * 100
final_val_acc = history.history["val_accuracy"][-1] * 100
print(f"Final Training Accuracy: {final_train_acc:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
