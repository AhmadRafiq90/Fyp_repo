import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
import datetime
import os
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def load_datasets(data_dir, img_size=(224, 224), batch_size=16, validation_split=0.2):
    """
    Loads training and validation datasets from a directory.
    The directory should have subdirectories for each class.
    label_mode is set to 'categorical' to produce one-hot encoded labels.
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset='training',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset='validation',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    return train_ds, val_ds

def data_augmentation_layer():
    """
    Returns a data augmentation pipeline using Keras layers.
    This pipeline rescales images and applies random flips, rotations, and zoom.
    """
    return tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

def build_pretrained_efficientnet_model(input_shape, num_classes):
    """
    Builds a model using the EfficientNetB0 architecture with pretrained ImageNet weights.
    The base model is frozen initially and a custom classifier head is added.
    """
    inputs = tf.keras.Input(shape=input_shape)
    # Apply data augmentation and rescaling
    x = data_augmentation_layer()(inputs)
    
    # Load the EfficientNetB0 model pre-trained on ImageNet
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False  # Freeze the base model for initial training
    
    # Add a global pooling layer, dropout, and a final classification head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def compute_class_weights(train_ds):
    """
    Computes class weights based on the distribution of classes in the training dataset.
    """
    labels = []
    for images, labs in train_ds:
        # labs is one-hot encoded; convert to integer labels
        labels.extend(np.argmax(labs.numpy(), axis=1))
    labels = np.array(labels)
    classes = np.unique(labels)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def plot_history(history):
    """
    Plots training and validation accuracy and loss over epochs.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, dataset, class_names):
    """
    Evaluates the model on a given dataset and prints the confusion matrix and classification report.
    """
    y_true = []
    y_pred = []
    for images, labels in dataset:
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=class_names)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

def main():
    # Update this path to point to your dataset directory.
    data_dir = "/media/ahmad/New Volume/potential datasets/archive_2/plantvillage dataset/color/"
    img_size = (224, 224)
    batch_size = 16
    epochs = 100  # Adjust based on your requirements
    
    # Load datasets
    train_ds, val_ds = load_datasets(data_dir, img_size, batch_size)
    num_classes = len(train_ds.class_names)
    print("Detected classes:", train_ds.class_names)
    
    # Compute class weights to handle imbalance
    cw = compute_class_weights(train_ds)
    print("Computed class weights:", cw)
    
    # Build the model using pretrained EfficientNetB0
    model = build_pretrained_efficientnet_model(input_shape=img_size + (3,), num_classes=num_classes)
    model.summary()  # Print the model architecture
    
    # Setup callbacks: checkpointing, learning rate reduction, and early stopping.
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f"./checkpoints/efficientnet_pretrained_{current_time}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "efficient_net__model.keras"),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=cw,
        callbacks=[checkpoint_callback, reduce_lr, early_stopping]
    )
    
    # Plot training history
    plot_history(history)
    
    # Evaluate the model on the validation set
    evaluate_model(model, val_ds, train_ds.class_names)
    
    # Save the final model
    model.save("final_pretrained_efficientnet_model.h5")
    print("Training complete. Final model saved as final_pretrained_efficientnet_model.h5")
    
if __name__ == "__main__":
    main()
