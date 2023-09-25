from PIL import Image
import os
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
import cv2
import h5py
import pickle
import matplotlib.pyplot as plt

class Preprocessing:
    def __init__(self):
        pass

    def create_preprocessed_dataset(self):
        # Define Folders Dataset path
        dataset_path = 'Dataset_Layer_3/test'

        # Define the target size (224x224)
        target_size = (224, 224)

        # Create an HDF5 file to store the images and labels
        output_hdf5_file = 'Dataset_Layer_3/Layer_three_data_test.h5'

        # Create a dictionary to store the mapping between class names and one-hot encoded labels
        class_mapping = {}

        # Create arrays to store images and labels
        images = []
        labels = []

        # Create a label encoder to convert class names to integer labels
        label_encoder = LabelEncoder()

        # Loop through the classes in the dataset
        for class_folder in os.listdir(dataset_path):
            # Check if the class folder is "Cards" and skip it
            """
            if class_folder == "Cards":
                continue

            if class_folder == "To-Subtitue":
                continue
            """

            if class_folder == ".DS_Store":
                continue

            class_path = os.path.join(dataset_path, class_folder)

            # Add class name to the list
            class_name = class_folder

            print(class_name)

            # Initialize the counter before the inner loop
            counter = 0 

            # Loop through the images in the class folder
            for image_file in os.listdir(class_path):
                # Check if the file is a directory, a file with an image extension, and not named ".DS_Store"
                if (
                    os.path.isfile(os.path.join(class_path, image_file)) and
                    image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and
                    image_file != ".DS_Store" #and
                    #counter <= 2500
                ):
                    image_path = os.path.join(class_path, image_file)

                    print(image_file)

                    # Open the image using Pillow
                    img = cv2.imread(image_path)

                    # Resize the image to the target size
                    img = cv2.resize(img, (target_size))

                    img = np.array(img)

                    img = img.astype('float32') / 255.0

                    # Append the image to the images array
                    images.append(img)

                    # Append the class name to the labels array
                    labels.append(class_name)

                    counter = counter + 1
                
                """
                if counter >= 2500:
                    break
                """
                

        images = np.array(images)
        print(images.shape)

        # Fit the label encoder on class names and transform them to integer labels
        integer_labels = label_encoder.fit_transform(labels)

        # Convert integer labels to one-hot encoded labels
        one_hot_labels = to_categorical(integer_labels, num_classes=len(label_encoder.classes_))

        # Create an HDF5 file
        with h5py.File(output_hdf5_file, 'w') as hdf5_file:
            hdf5_file.create_dataset('images', data=images, compression='gzip')
            hdf5_file.create_dataset('labels', data=one_hot_labels, compression='gzip')

        # Save the class mapping to a separate file using pickle
        class_mapping_file = 'Dataset_Layer_3/Layer_three_data_test.pkl'
        with open(class_mapping_file, 'wb') as mapping_file:
            class_mapping = dict(zip(labels, one_hot_labels.tolist()))
            pickle.dump(class_mapping, mapping_file)

        print("Images resized and saved to HDF5 file with one-hot encoded labels and class mapping successfully.")

        # Path to the HDF5 file
        output_hdf5_file = 'Dataset_Layer_3/Layer_three_data_test.h5'

        # Path to the class mapping file
        class_mapping_file = 'Dataset_Layer_3/Layer_three_data_test.pkl'

        # Open the HDF5 file in read mode
        with h5py.File(output_hdf5_file, 'r') as hdf5_file:
            # Get the shapes of the datasets
            images_shape = hdf5_file['images'].shape
            labels_shape = hdf5_file['labels'].shape

            print(f"Images dataset shape: {images_shape}")
            print(f"Labels dataset shape: {labels_shape}")

        # Load and print the class mapping from the pickle file
        with open(class_mapping_file, 'rb') as mapping_file:
            class_mapping = pickle.load(mapping_file)

            print("\nClass Mapping:")
            for class_name, one_hot_label in class_mapping.items():
                print(f"Class Name: {class_name}, One-Hot Label: {one_hot_label}")

        # Load the HDF5 file and retrieve an image
        with h5py.File(output_hdf5_file, 'r') as hdf5_file:
            images = hdf5_file['images'][:]

        # Choose an index to select an image
        index = 0  # You can change this to the index of the image you want to visualize

        # Get the selected image
        image = images[index]

        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis('off')  # Turn off the axis
        plt.show()


