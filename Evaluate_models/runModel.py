import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
#model = tf.keras.models.load_model('/mnt/e/Final Year Project/Python/Data_Preprocessing/Fyp_repo/checkpoints/15sp_74.9pacc/model.keras')
model = tf.keras.models.load_model('best_model.keras')
# Directory containing test data
test_data_dir = '/mnt/e/potential datasets/archive_2/plantvillage dataset/test/'

# Image data generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create a test data generator
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),  # Adjust to your model's input size
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Predict the classes
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# True classes
true_classes = test_generator.classes
# Calculate confidence scores for each category
confidence_scores = np.max(predictions, axis=1)

# Print confidence scores for each category
# for i, score in enumerate(confidence_scores):
#     print(f'Image {i+1}: Confidence Score: {score:.2f}')
# Calculate accuracy
accuracy = accuracy_score(true_classes, predicted_classes)
print(f'Overall Accuracy: {accuracy * 100:.2f}%')

# Classification report
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)