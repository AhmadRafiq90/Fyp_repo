import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
mixed_precision.set_global_policy("mixed_float16")

# Define Data Augmentation Layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2)
])

# Define Patch Embedding Layer
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = layers.Dense(embedding_dim)
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        patch_dim = self.patch_size * self.patch_size * x.shape[-1]  # Flattened patch size
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        num_patches = patches.shape[1] * patches.shape[2]  # Compute dynamically
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dim])
        return self.projection(patches)

# Define Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)

# Build Vision Transformer Model
def build_vit(input_shape=(224, 224, 3), num_classes=39, patch_size=4, embed_dim=128, num_heads=4, ff_dim=256, num_layers=6):
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    x = PatchEmbedding(patch_size, embed_dim)(augmented)
    
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model



# Create the model
vit_model = build_vit()
vit_model.summary()

# Function to load images from a folder with validation split
def load_images_from_folder(folder_path, image_size=(224, 224), batch_size=6, validation_split=0.2):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        folder_path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        subset="training",
        seed=123
    )
    
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        folder_path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        subset="validation",
        seed=123
    )
    
    return train_dataset, val_dataset

# Load training and validation datasets
train_dataset, val_dataset = load_images_from_folder("/mnt/e/potential datasets/archive_2/plantvillage dataset/color/")

# Apply data augmentation to training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# Compile the model
vit_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model with validation data
vit_model.fit(train_dataset, validation_data=val_dataset, epochs=10)
