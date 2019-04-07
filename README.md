# Self-Supervised Classification
### Python code for self-supervised classification of remotely sensed imagery - part of the Deep Riverscapes project
Supervised classification is a workflow in Remote Sensing (RS) whereby a human user draws training (i.e. labelled) areas, generally with a GIS vector polygon, on a RS image.  The polygons are then used to extract pixel values and, with the labels, fed into a supervised machine learning algorithm for land-cover classification.  The core idea behind *Self-Supervised* Classification (SSC) is to replace the human user with a pre-trained convolutional neural network (CNN).    Once a CNN is trained, SSC starts by running the trained CNN on an image.  This results in a tiled image classifation.  Then SCC runs a second phase where the CNN-derived tiled classification is used to train and run a more shallow machine learning algorithm but only on the image pixels of that given image making the result more customised to the specific radiometric properties of the image.   The output is a pixel-level clasification for land-cover.  We have experimented with Random Forests and Multi Layer Perceptrons (MLP) and found that the MLP gives better results.  Our test dataset is compiled from high resolution aerial imagery of 11 rivers.  It has 1 billion labelled pixels for training and another 4 billion labelled pixels for validation.  If we train 11 CNN models, 1 for each river, then validate these CNN models only with the validation images of their repective rivers, we obtain an overall pixel-weighted F1 score of **94%**.  If we train a single CNN with the data from 5 rivers, we find that the resulting SSC workflow can predict classes of the *other* 6 rivers (true out of sample data never seen during CNN training) with an overall pixel-wieghted F1 sore of **87%**. See citation below.

 ## Dependencies
* Keras (we use TensorFlow-GPU as the backend)
* Scikit-Learn
* Imbalanced-Learn toolbox 
* Scikit-Image
* Pandas

## Getting Started

### Disclaimer
This code is currently in the development stage and inteneded for research purposes.  The coding structure is naive and not optimised for production.  The process is not yet designed to output class rasters for new unclassified images and expects every image to have an accompanying class raster for either training or for validation. 

### Data preparation
It is assumed that the data comes in the format that typically results from an airborne survey such as: root_number.  We recommend that the data be structured as: RiverName_Number.jpg.  The number must be in 5 digit format.  The associated classification is expected to have the same filename but with a prefix of 'SCLS' and a tif format.  

### CNN Training
Once image data is organised, the first script should be edited.  User options are at the start.  It is recommended to set the ModelTuning variable to True in the first instance and run the tuning procedure for the CNN.  This will output a figure and the correct number of tuning epochs can be set as the point where the loss and accuracy of the validation data begin to diverge from the loss and accuracy of the training data.  Once this is established, the script must be run again with ModelTuning set to False and the correct value for Tuning. This will save the model with a .h5 extension and it will also save a class key as a small csv file.

### SSC execution
Once a trained CNN model is in place, SSC performance can be evaluated.  The images to test must follow the same naming convention and all have an existing set of manual labels used to calculate validation statistics.  These should not be the same images as used to train the CNN.  Once the numerous parameters have been adjusted (seee citation), the script will execute and output performance metrics for each image.  csv files with a CNN_ prefix give performance metrics for the CNN model with F1 scores and support (# of pixels) for each class.  SSC_ files give the same metrics for the final SSC result after the application of the MLP or the Random Forest selected in the options. A 4-part figure will also be output showing the original image, the existing class labels, the CNN classification and the final SSC classification labelled either MLP or RF.




## Authors:
 - Patrice E. Carbonneau, University of Durham, [e-mail](mailto:patrice.carbonneau@durham.ac.uk)
 - Toby P. Breckon, University of Durham
 - James T. Dietrich, University of Northern Iowa
 - Steven J. Dugdale, University of Nottingham
 - Mark A. Fonstad, University of Oregon
 - Hitoshi Miyamoto, Shibaura Institute of Technology
 - Amy S. Woodget, Loughborough University
 
 ## Citation
 This work is currently in the process of publication where a full description of parameters will be available.  The current best citation is:
 
 
 

