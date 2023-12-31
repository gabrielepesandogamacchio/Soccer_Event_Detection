# Soccer Event Detection
**Report: [Soccer Event Detection Report](https://drive.google.com/file/d/11IvTDSavQrdbyP99LZB2ycUywTgInRkh/view?usp=sharing)** 

This project focuses on the classification of soccer events using the SEV dataset provided by the research Soccer Event Detection Using Deep Learning. Building upon the state-of-the-art methodology described in the previous study, we constructed a pipeline for image classification. Notably, the dataset was dissected into sub-events to enhance classification accuracy. While the efficientNetB0 architecture demonstrated its effectiveness in improving classification results over the test set, our approach explored the use of a custom architecture.

In this report, we present the results achieved with our custom architecture, acknowledging that they may not surpass the state-of-the-art benchmarks. Furthermore, we delve into insightful observations regarding the test set and the model evaluation process. Despite our rigorous implementation of the proposed architecture and pipeline, we encountered challenges in reproducing the results reported in the referenced research.

![NeuralNetwork](https://github.com/gabrielepesandogamacchio/Soccer_Event_Detection/blob/main/Soccer_Event_Detection_Project/NeuralNet.png)

## Table of Contents
- [Dataset](#dataset)
- [Usage](#usage)
- [Documentation](#documentation)


## Dataset

Access the dataset from the following Google Drive folder: 
- [Official Dataset](https://drive.google.com/drive/folders/1jzt7g0KqFNTshEAau95aScPWin55g31E)
- [Preprocessed Dataset](https://drive.google.com/drive/folders/1dUSMI8Rn5jfezskXHCFucJvxeATKenjA?usp=sharing)

In the first Google Drive folder, you can find:

1. Raw Image Data stored in the:
- `Train` folder
- `Test` folder
- `Soccer` folder
- `Event` folder

2. Preprocessed data saved in h5 file extension available here:
- First, second, third and final Layer Dataset: [h5 preprocessed dataset](https://drive.google.com/drive/folders/1dUSMI8Rn5jfezskXHCFucJvxeATKenjA?usp=sharing)
These preprocessed datasets have been used for training and testing the 4 models.

After downloading these files, place them into the `Data` folder of the project and update the path in the code to load the dataset correctly.

## Usage
Within the project, you'll come across the main scripts: 
  - main.py
Running the main allows you to test a specific model. If you want to generate the preprocessed dataset from the raw images run the preprocessing class. If you want to test a specific model (custom CNN or EfiicientNetB0) comment the model you don't want to build and compile.

## Documentation
For more details on the project's implementation and usage, you can read the the **Report: [Soccer Event Detection Report](https://drive.google.com/file/d/11IvTDSavQrdbyP99LZB2ycUywTgInRkh/view?usp=sharing)**
