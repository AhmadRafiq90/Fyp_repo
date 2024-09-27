import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

#Function to Load and Preprocess the Data
def load_data(train_dir, img_size = (224, 224), batch_size = 32, validation_split = 0.2):
    "ImageGenerator for Data Augmentation and rescaling"
    datagen = ImageDataGenerator(
        rescale = 1./255,                       #Normalizing to [0,1]"
        validation_split = validation_split     #20 % of data is reserved for testing(validation)
    )
    
    "Training data generator"
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size = img_size,                 #Resizing image to 224 X 224
        batch_size = batch_size,                #Batch size for training
        class_mode = 'categorical',            #For multi-class classification
        subset = 'training'                     #Training data
    )
    
    "Validation data generator"
    validation_generator = datagen.flow_from_directory(
        train_dir, 
        target_size = img_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        subset = 'validation'                   #Testing (Validation) data
    )
    
    return train_generator, validation_generator
    
"Function to build the CNN model"
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
        layers.Dense(num_classes, activation='softmax')                           #Output Layer has number of neurons equal to num_classes using softmax activation function to produce
                                                                                  #probability distribution over classes
    ])
    return model

"Function to train the model"
def train_model(model, train_generator, validation_generator, epochs = 10):
    "epochs refers to number of complete passes through the entire training dataset with 1 epochs meaning that every sample in the training dataset has been used once to update weights of the model"
    "During each epoch, model processes the training data and computes the loss function and updates the weights according to the loss"
    
    "Compile the model"
    "opimizer = adam -> adam is a popular optimizer that adapts the learning rate during training"
    "categorical_crossentropy -> it is used for multi-classification problems measuring how well the predicticted probabilities match the true class labels"
    "metrics = ['accuracy'] -> helps you measure model's accuracy during training"
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    "Now Train the Model"
    history = model.fit(
        train_generator,
        validation_data = validation_generator,
        epochs = epochs
    )
    
    "history contains the information about the training process including loss and accuracy of each epochs"
    return history

"Main Function to execute the training Pipeline"
def main():
    "Set the directory for training images"
    train_dir = "G:/Final Year Project/Dataset/Selected_Species_train"
    
    "Load the data"
    train_generator, validation_generator = load_data(train_dir)
    
    "Get input shape and number of classes"
    input_shape = (224, 224, 3)   #3 represents the color chanels here i.e (R,G,B)
    num_classes = len(train_generator.class_indices)  #Counts the number of unique classes based on subfolder structure in the "train_dir", This is cruicial for configuring output layer of the model
    
    "Build the model"
    model = build_model(input_shape, num_classes)
    
    "Train the model"
    history = train_model(model, train_generator, validation_generator, epochs = 10)
    
    #Printing Training and validation accuracy
    print(f"Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    
    "Saved the trained model"
    model.save("species_classifier_model.keras")   #After Training the model It can be safed to a file and can be loaded later on.
    
if __name__ == "__main__":
    main()
    
    
    
    