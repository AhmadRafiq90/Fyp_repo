import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime

def load_data(train_dir,img_size = (224, 224), batch_size = 8, validation_split = 0.2):
    "ImageGenerator for Data Augmentation and rescaling"
    datagen = ImageDataGenerator(
        rescale = 1./255,                          #Normalizing to [0,1]"
        validation_split = validation_split,     #20 % of data is reserved for testing(validation)
        rotation_range=40,  # Randomly rotate images by up to 40 degrees
        width_shift_range=0.2,  # Shift images horizontally by 20%
        height_shift_range=0.2,  # Shift images vertically by 20%
        shear_range=0.2,  # Shear transformation
        zoom_range=0.2,  # Random zoom on images by 20%
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest',  # Fill pixels after transformations
    )
    
    "Training data generator"
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size = img_size,                 #Resizing image to 224 X 224
        batch_size = batch_size,                #Batch size for training
        class_mode = 'categorical',            #For multi-class classification
        subset = 'training'                     #Training data
    )
    # After loading the data
    #print("Class Indices:", train_generator.class_indices)
    "Validation data generator"
    validation_generator = datagen.flow_from_directory(
        train_dir, 
        target_size = img_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        subset = 'validation'                   #Testing (Validation) data
    )
    
    return train_generator, validation_generator

    
"Function to build the CNN model"#
def build_model(input_shape, num_classes):  #Input_shape = (height, width, channels etc), number of output class for classification
    "Conv2D -> Resposible for feature extraction"
    "MaxPooling2D -> These layers downsample feature maps while retaining important features"
    "Flatten -> Converts 2D feature maps to 1D vector which is necessary before feedding the data into a fully connected layer"
    "Dense -> Fully Connected Layers"
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape = input_shape),  #First Layer having 32 filters os size 3 X 3, using RELU activation function
        layers.MaxPooling2D((2, 2)),                       
        layers.Conv2D(64, (3, 3), activation='relu'),                             #Second Layer having 64 filters
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),                            #Third Layer having 128 filters
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),   
        layers.Dense(128, activation='relu'),                                     #128 neurons and uses the RELU activation function
        #layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')                           #Output Layer has number of neurons equal to num_classes using softmax activation function to produce
                                                                                  #probability distribution over classes
    ])
    return model

from tensorflow.keras.callbacks import EarlyStopping

"Function to train the model with Early Stopping"
def train_model(model, train_generator, validation_generator, epochs, patience, checkpoint_dir):
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Define checkpoint filepath
    checkpoint_filepath = os.path.join(checkpoint_dir, "manual_deletion_model.keras")
    
    # Check if a checkpoint exists and load the model weights from the last saved checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        #model.load_weights(latest_checkpoint)
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    # Callback to save the model every 10 epochs
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,  # Save path
        monitor='val_accuracy',
        mode = 'max',
        save_best_only=True,
        save_weights_only=False,  # Save only the model weights (not the entire model)
        verbose=1  # Verbose output for checkpoint saving
    )
    """
    epochs refers to number of complete passes through the entire training dataset.
    Early stopping stops training if the validation loss doesn't decrease for 'patience' number of epochs.
    """
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy',       # Monitors validation loss
        patience=patience,        # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity
    )
    
    # Train the model with EarlyStopping
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint_callback]  # Add early stopping callback
    )
    
    return history

def main():

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_shape = (224, 224, 3)  # 3 represents the color channels (R, G, B)
    
    "Set the directory for training images"
    #train_dir = "/media/ahmad/New Volume/potential datasets/archive2/train/"
    checkpoint_dir = f"./checkpoints/model_{current_time}"
    train_dir = "/mnt/d/potential datasets/archive_2/plantvillage dataset/color/" 
    #val_dir = "/media/ahmad/New Volume/potential datasets/archive2/val/"
    #val_dir = "/mnt/e/potential datasets/archive_2/plantvillage dataset/"
    "Load the data"
    train_generator, validation_generator = load_data(train_dir)
    
    "Get input shape and number of classes"

    num_classes = len(train_generator.class_indices)  # Determine the number of classes from directory structure

    "Build the model"
    model = build_model(input_shape, num_classes)
    
    "Train the model with early stopping"
    history = train_model(model, train_generator, validation_generator, epochs=100, patience=40, checkpoint_dir=checkpoint_dir)
    
    # Printing training and validation accuracy
    print(f"Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    
    "Save the trained model"
    model.save("40sp_test.keras")
    #model.save(f"{history.history['val_accuracy'][-1] * 100:.2f}%.keras")
    
if __name__ == "__main__":
    main()
 