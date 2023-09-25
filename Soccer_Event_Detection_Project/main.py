from src.Preprocessing import Preprocessing
from src.CustomCNN import CustomCNN
from src.EfficientNetB0Model import EfficientNetB0Model

if __name__ == "__main__":
    # Step 1 -> Preprocessing - Build Dataset h5 files
    """
        in the preprocessing class change the directory path in
        order to process the correct dataset directory. Also change the 
        name of the h5 files that will be generated.
    """
    preproccesing = Preprocessing()
    preproccesing.create_preprocessed_dataset()

    """
        choose the model to be trained:
         -Custom CNN -> step 2a
         -EfficientNetB0 -> step 2b
    """

    # Step 2a -> Train the custom CNN model and save the results
    custom_cnn_model = CustomCNN()
    custom_cnn_model.build_and_run_model()

    # Step 2b -> Train the EfficientNetB0 model and save the results
    effNet_model = EfficientNetB0Model()
    effNet_model.build_and_run_model()



