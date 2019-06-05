### Self-Supervised Classification Sample Data

The data in this folder is mainly intended to allow users verify that the code functions correctly on their system without the need to download more data from the larger data repository.

1) in the 'Train' Folder, there are samples of images/label pairs for training (with representative filenames)
    - Downloade this data and on line 83 of TrainCNN.py, enter the correct file path.
    
2) in the model folder `NASNetM_ImNet_50x.h5` is a pre-trained NASNet Mobile model with weights initialised with the ImageNet database.  The model is set to work with image tiles of 50X50 pixels.  Download this model to the same folder as the training images above and enter the name (without the .h5 extension) on line 86 of TrainCNN.py

3) run TrainCNN.py.  Monitor your GPU usage and insure that TensorFlow-GPU is working correctly.

4) in the 'Validate' folder, there are sample images that can be used for a mock validation of the trained model. Download these to a new folder and edit lines 79 to 83 of `SelfSupervisedClassification.py`  

5) At the end, the folder designated as the ScorePath on line 82 of `SelfSupervisedClassification.py`  should have 10 4-part figures in png format and 20 small csv files.  Use CompileClassiticationReport to merge the csv files into a single database formatted for use with the Seaborn visualisation library.


   
