### Self-Supervised Classification Sample Data

The data in this folder is intended to allow users to verify that the code functions correctly on their system without the need to download the large data volume from the repository.

1) in the 'Train' Folder, there are samples of images/label pairs for training (with representative filenames). Please note that this small sample was not selected to produce a well trained CNN model.  The only purpose of the sample data is for testing and debugging. 
    - Downloade this data and on line 84 of TrainCNN.py, enter the correct file path.
    
2) Also in the 'Train' folder, `NASNet_Mobile_50px.h5` is a base NASNet Mobile model with weights initialised with the ImageNet database.  The model is set to work with image tiles of 50X50 pixels.  Download this model to the same folder as the training images. By default, this name (without the .h5 extension) is on line 86 of TrainCNN.py.  Enter a different name on line 87 for the final trained version of the CNN model.

3) run TrainCNN.py.  Monitor your GPU usage and insure that TensorFlow-GPU is working correctly.

4) in the 'Validate' folder, there are sample images that can be used for a mock validation of the trained model. Download these to a new folder and edit lines 79 to 83 of `SelfSupervisedClassification.py`.  Line 79 should have the model name selected in TrainCNN.py.  Line 80 should point to the same file path as line 84 in TrainCNN.py.  The other parametes can be left as dafault in the first instance.  

5) At the end, the folder designated as the ScorePath on line 82 of `SelfSupervisedClassification.py`  should have 10 4-part figures in png format and 20 small csv files.  Use CompileClassiticationReport to merge the csv files into a single database formatted for use with the Seaborn visualisation library.


   
