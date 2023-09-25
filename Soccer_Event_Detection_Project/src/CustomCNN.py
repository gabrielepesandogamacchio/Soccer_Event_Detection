from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class CustomCNN:
    def __init__(self):
        pass

    def build_and_run_model(self):
        dataset_path = 'src/dataset_soccer_event.h5'

        # Load Dataset
        with h5py.File(dataset_path, 'r') as hdf5_file:
            images = hdf5_file['images'][:]
            labels = hdf5_file['labels'][:]

        X, y = shuffle(images, labels, random_state=1)

        #X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.08)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

        #inpect the shape of the training and testing.
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)
        print(X_test.shape)
        print(y_test.shape)

        NUM_CLASSES = 2
        IMG_SIZE = 224
        size = (IMG_SIZE, IMG_SIZE)

        def build_model(num_classes):
            inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

            # Define the model layers
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3))(inputs)
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(4096, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(4096, activation='relu')(x)

            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

            # Create the model
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SimpleCNN")

            # Compile the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(
                optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
            )

            return model

        model = build_model(num_classes=NUM_CLASSES)

        # Define data augmentation parameters
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,          # Rotate images by up to 30 degrees
            width_shift_range=0.05,     # Shift images horizontally by up to 20% of the width
            height_shift_range=0.05,    # Shift images vertically by up to 20% of the height
            zoom_range=0.05,            # Zoom in/out by up to 20%
            horizontal_flip=False,      # Flip images horizontally (left to right)
            vertical_flip=False,        # Flip images vertically (top to bottom)
        )

        # use the data generator to apply data augmentation
        hist = model.fit(
            datagen.flow(X_train, y_train, batch_size=2),
            epochs=4,
            validation_data=(X_val, y_val)
        )

        #______________ PLOT  MODEL History ___________________

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

        #______________ SAVE MODEL ___________________

        # Replace 'my_model' with your model variable name
        tf.saved_model.save(model, 'src/NN_soccer_event_classification')


        #______________ MODEL EVALUATION _____________

        # evaluate the model on the test set for this fold
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Fold {0} test loss: {test_loss:.4f}")
        print(f"Fold {0} test accuracy: {test_acc:.4f}")

        # generate predictions on the test set for this fold
        y_pred = np.argmax(model.predict(X_test), axis=-1)