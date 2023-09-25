from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class EfficientNetB0Model:
    def __init__(self):
        pass

    def build_and_run_model(self):
        dataset_path = 'src/dataset_soccer_event.h5'

        # Load Dataset
        with h5py.File(dataset_path, 'r') as hdf5_file:
            images = hdf5_file['images'][:]
            labels = hdf5_file['labels'][:]

        X, y = shuffle(images, labels, random_state=1)

        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2 )

        #inpect the shape of the training and testing.
        print(X_train.shape)
        print(y_train.shape)
        print(X_validation.shape)
        print(y_validation.shape)

        NUM_CLASSES = 9
        IMG_SIZE = 224

        def build_model(num_classes):
            inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

            # EfficientNetB0 Layer
            model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

            # Freeze the pretrained weights
            model.trainable = False

            # Remove Global Average Pooling layer
            x = tf.keras.layers.GlobalAveragePooling2D(name="max_pool")(model.output)

            # Flatten the output of the EfficientNet model
            x = tf.keras.layers.Flatten()(model.output)

            # First Dense-512 layer
            x = tf.keras.layers.Dense(4096, activation="relu", name="dense_512_1")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Dropout(0.5)(x)

            # Second Dense-512 layer
            x = tf.keras.layers.Dense(4096, activation="relu", name="dense_512_2")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            # Dense-10 layer
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

            # Compile
            model = tf.keras.Model(inputs, outputs, name="EfficientNet")
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(
                optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
            )

            model.summary()

            return model

        model = build_model(num_classes=NUM_CLASSES)

        # Define your data augmentation parameters
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,          # Rotate images by up to 30 degrees
            width_shift_range=0.05,     # Shift images horizontally by up to 20% of the width
            height_shift_range=0.05,    # Shift images vertically by up to 20% of the height
            zoom_range=0.05,            # Zoom in/out by up to 20%
            horizontal_flip=False,      # Flip images horizontally (left to right)
            vertical_flip=False,        # Flip images vertically (top to bottom)
        )

        # Now, when you train the model, use the data generator to apply data augmentation
        hist = model.fit(
            datagen.flow(X_train, y_train, batch_size=2),
            epochs=3,
            validation_data=(X_validation, y_validation)
        )

        # Customize the plot
        plt.figure(figsize=(12, 6), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(hist.history['accuracy'], linestyle='-', color='b', label='Train')
        plt.plot(hist.history['val_accuracy'], linestyle='--', color='r', label='Validation')
        plt.title('Model Accuracy Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(hist.history['loss'], linestyle='-', color='b', label='Train')
        plt.plot(hist.history['val_loss'], linestyle='--', color='r', label='Validation')
        plt.title('Model Loss Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        plt.show()
