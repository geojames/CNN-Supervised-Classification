#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'
__date__ = '15 APR 2019'
__version__ = '1.1'
__status__ = "initial release"
__url__ = "https://github.com/geojames/Self-Supervised-Classification"

"""
Name:           TrainCNN.py
Compatibility:  Python 3.6
Description:    This file trains and outputs a NASNet Large model to for the 
                task of river classification from RGB images

Requires:       keras, TensorFlow-GPU, numpy, pandas, matplotlib, skimage, sklearn

Dev Revisions:  JTD - 19/4/19 - Updated file paths, added auto detection of
                    river names from input images, improved image reading loops

Licence:        MIT
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""

from keras import layers
from keras import models
from keras import regularizers
from keras import optimizers
from keras.applications import nasnet
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
import os.path
import sys
from sklearn import metrics
import copy
from skimage.morphology import disk
from skimage.filters.rank import median, entropy
from IPython import get_ipython
import glob
import fnmatch

#########################################################
#Image loading and data preparation.
#This loads all images, and class label masks in the TRAIN folder and it:
# 1 - Tiles the images into a 4D tensor of (nTiles, Width, Height, Bands), channel-last format
# 2 - Looks at the classification mask and when a tile is 90% classified, produces a label vector value. 0 if less than 90%
# 3 - Feeds this into the NASNet Large convnet.  Level of trainability of the convnet can be set by user

#------------------------------------------------------------
"""User data input. Fill in the info below before running"""
#------------------------------------------------------------
#Watch the \\ and if there is a bug go to single quotes.  This folder shold have the training images, labels and the pre-trained NASNet model.
#Trained model and class key will also be written out to the training folder
TrainPath = 'E:\\DeepRiverscapes\\MargSix'  
#Input and Output name for the model.  It wil be saved in the training folder
ModelInputName = 'NASNetLarge_base_50px' #no extensions
ModelOutputName = 'Marg6'      
         #where the model will be saved

NClasses = 5 #The number of classes in the imagery. Adjust as needed



#use this option to train the model with a validation subset. Accuracy and loss checks will be displayed.
#When the tuning is satisfactory, set to False and train with the whole dataset. Model will only be saved if this is set to False
#When true the sript will exit with a system error and display loss/acc vs epoch figures.  This is intentional.
ModelTuning = False
TuningEpochs = 20 
TuningSubSamp = 0.55 # Subsample of data, 0-1, to be used in tuning.

#If the model is tuned, enter the right number of epochs.
#This is only used when ModelTuning is False.  
TrainingEpochs = 7
BatchSize = 100 #will depend on your GPU

# Path checks- checks for folder ending slash, adds if nessesary
if ('/' or "'\'") not in TrainPath[-1]:
    TrainPath = TrainPath + '/'



##################################################################
""" HELPER FUNCTIONS SECTION"""
##################################################################
# Helper function to crop images to have an integer number of tiles. No padding is used.
def CropToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]
    
    
#Helper functions to move images in and out of tensor format
def split_image_to_tiles(im, size):
    
    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape

     
    nTiles_height = h//size
    nTiles_width = w//size
    TileTensor = np.zeros((nTiles_height*nTiles_width, size,size,d))
    B=0
    for y in range(0, nTiles_height):
        for x in range(0, nTiles_width):
            x1 = np.int32(x * size)
            y1 = np.int32(y * size)
            x2 = np.int32(x1 + size)
            y2 = np.int32(y1 + size)
            TileTensor[B,:,:,:] = im[y1:y2,x1:x2].reshape(size,size,d)
            B+=1

    return TileTensor

#Create the label vector
def PrepareTensorData(ImageTile, ClassTile, size):
    #this takes the image tile tensor and the class tile tensor
    #It produces a label vector from the tiles which have 90% of a pure class
    #It then extract the image tiles that have a classification value in the labels
    LabelVector = np.zeros(ClassTile.shape[0])
    
    for v in range(0,ClassTile.shape[0]):
        Tile = ClassTile[v,:,:,0]
        vals, counts = np.unique(Tile, return_counts = True)
        if (vals[0] == 0) and (counts[0] > 0.1 * size**2): #Handle unlabelled (class = 0) patches
            LabelVector[v] = 0
        elif counts[np.argmax(counts)] >= 0.9 * size**2:
            LabelVector[v] = vals[np.argmax(counts)] 
    
    ClassifiedTiles = np.zeros((np.count_nonzero(LabelVector), size,size,3))
    ClassifiedLabels = np.zeros((np.count_nonzero(LabelVector),1))
    C = 0
    for t in range(0,len(LabelVector)):
        if LabelVector[t] > 0:
            ClassifiedTiles[C,:,:,:] = ImageTile[t,:,:,:]
            ClassifiedLabels[C,0] = LabelVector[t]
            C += 1
    return ClassifiedLabels, ClassifiedTiles


#############################
def class_prediction_to_image(im, PredictedTiles, size):

    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape

     
    nTiles_height = h//size
    nTiles_width = w//size
    #TileTensor = np.zeros((nTiles_height*nTiles_width, size,size,d))
    TileImage = np.zeros(im.shape)
    B=0
    for y in range(0, nTiles_height):
        for x in range(0, nTiles_width):
            x1 = np.int32(x * size)
            y1 = np.int32(y * size)
            x2 = np.int32(x1 + size)
            y2 = np.int32(y1 + size)
            TileImage[y1:y2,x1:x2] = np.argmax(PredictedTiles[B,:])
            B+=1

    return TileImage


    
################################################################################
#Keep track of the new classes in a Pandas DataFrame
def MakePandasKey(ClassKey, NClasses, f, r):
    for c in range(1,NClasses+1):
        ClassKey['River'].loc[c+(f*NClasses)] = r
        ClassKey['LocalClass'].loc[c+(f*NClasses)] = (f*NClasses+1) + (c-1)
        ClassKey['HierarchClass'].loc[c+(f*NClasses)] = 1 + (c-1)
        if c==1:
            ClassKey['ClassName'].loc[c+(f*NClasses)] = 'water'
        elif c==2:
            ClassKey['ClassName'].loc[c+(f*NClasses)] = 'sediment'
        elif c==3:
            ClassKey['ClassName'].loc[c+(f*NClasses)] = 'green_vegetation'
        elif c==4:
            ClassKey['ClassName'].loc[c+(f*NClasses)] = 'senescent_vegetation'
        else:
            ClassKey['ClassName'].loc[c+(f*NClasses)] = 'Road'
    return ClassKey
            
###############################################################################
# Return a class prediction to the 1-Nclasses Macro Classes classes
def SimplifyClass(ClassImage, ClassKey):
    Iclasses = np.unique(ClassImage)
    for c in range(0, len(Iclasses)):
        KeyIndex = ClassKey.loc[ClassKey['LocalClass'] == Iclasses[c]]
        Hclass = KeyIndex.iloc[0]['HierarchClass']
        ClassImage[ClassImage == Iclasses[c]] = Hclass
    return ClassImage

###############################################################################
# Checks            
def checkInputImgNames(path):
    # Glob list fo all jpg images
    img = glob.glob(path+"*.jpg")

    # test file names using fnmatch fo a 5-digit number
    #   if aone false, raise an exception and kill everything 
    for im in img:
        if fnmatch.fnmatch(im,'*_[0-9][0-9][0-9][0-9]*.jpg') == False:       
            raise Exception("Image: %s is not in the correct format. All images need to have a five digit number after the RiverName, eg RiverName_00000.jpg"%(os.path.basename(im)))
            sys.exit(1)
    

###############################################################################
"""Data Preparation Section"""
###############################################################################    
size = 50 #Do not edit. The base models supplied all assume a tile size of 50.  

# Check Files Name Pattern
checkInputImgNames(TrainPath)

# Getting River Names from the files
# Glob list fo all jpg images, get unique names form the total list
img = glob.glob(TrainPath + "*.jpg")
RiverTuple = []
for im in img:
    RiverTuple.append(os.path.basename(im).partition('_')[0])
RiverTuple = np.unique(RiverTuple)

# Get training class images
class_img = glob.glob(TrainPath + "SCLS_*.tif*")

if len(img) != len(class_img):
    raise Exception("Training and Classified Images Counts do not match, Please Double-check and Rerun")
    sys.exit(1)


ClassKeyDict ={'LocalClass': np.zeros(1+(len(RiverTuple)*5)).ravel(), 'HierarchClass' : np.zeros(1+(len(RiverTuple)*5)).ravel()} 
ClassKey = pd.DataFrame(ClassKeyDict)
ClassKey['River'] = ""
ClassKey['ClassName'] = ""

LocalClasses = np.int16(NClasses*len(RiverTuple))  
ImageTensor = np.zeros((1,size,size,3))
LabelTensor = np.zeros((1,(NClasses*len(RiverTuple))+1))
   
for f,riv in enumerate(RiverTuple):
    ClassKey = MakePandasKey(ClassKey, NClasses, f, riv)
    for i,im in enumerate(img):  
        print('processing image ' + os.path.basename(im))
        Im3D = io.imread(im)
        if len(Im3D) == 2:
            Im3D = Im3D[0]

        Class = io.imread(class_img[i])
        if Im3D.shape[0] != Class.shape[0]:
            sys.exit("Error, Image and class mask do not have the same dimensions")
            
        NewClass = Class + f*NClasses #Transform macro-class from the classification 1 to N where N is the number of classes to micro-classes.
        NewClass[Class == 0] = 0 #this step avoids the unclassified areas becoming class f*(NClasses)
        Class = copy.deepcopy(NewClass)
        ImCrop = CropToTile (Im3D, size)
        ClsCrop =  CropToTile (Class, size)           
        I_tiles = split_image_to_tiles(ImCrop, size)
        I_tiles = np.int16(I_tiles) / 255
        C_tiles = split_image_to_tiles(ClsCrop, size)
        CLabelVector, ClassifiedTiles = PrepareTensorData(I_tiles, C_tiles, size)
        Label1hot= to_categorical(CLabelVector, num_classes = LocalClasses+1) #convert to one-hot encoding
        ImageTensor = np.concatenate((ImageTensor,ClassifiedTiles), axis = 0)
        LabelTensor = np.concatenate((LabelTensor, Label1hot), axis = 0)
        del(ImCrop,ClsCrop,I_tiles, C_tiles, Label1hot, Class, NewClass, Im3D, CLabelVector, ClassifiedTiles)
            

#Delete the first blank tile from initialisation
ImageTensor = ImageTensor[1:,:,:,:]
LabelTensor = LabelTensor[1:,:]
print(str(ImageTensor.shape[0]) + ' image tiles of ' + str(size) + ' X ' + str(size) + ' extracted')    
           

##########################################################
"""Convnet section"""
##########################################################
#Setup the convnet and add dense layers
print('Loading ' + ModelInputName + '.h5') #be sure the pre-trained CNN is in the same folder as the training images
FullModelPath = TrainPath + ModelInputName + '.h5'
conv_base = load_model(FullModelPath)


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_regularizer= regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(LabelTensor.shape[1], activation='softmax'))

#Freeze all or part of the convolutional base to keep imagenet weigths intact
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    set_trainable = False
    if  ('separable_conv_2_normal_left2' in layer.name) or ('separable_conv_2_normal_right2' in layer.name):
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

            

#Tune an optimiser
Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
 
#Compile and display the model          
model.compile(optimizer=Optim,loss='categorical_crossentropy', metrics=['acc'])
model.summary()


if ModelTuning:
    #Split the data for tuning. Use a double pass of train_test_split to shave off some data
    (trainX, testX, trainY, testY) = train_test_split(ImageTensor, LabelTensor, test_size=TuningSubSamp-0.001)
    (Partiel_trainX, Partiel_testX, Partiel_trainY, Partiel_testY) = train_test_split(testX, testY, test_size=0.25)
    del(ImageTensor, LabelTensor)
    history = model.fit(Partiel_trainX, Partiel_trainY, epochs = TuningEpochs, batch_size = 75, validation_data = (Partiel_testX, Partiel_testY))
    #Plot the test results
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    
    epochs = range(1, len(loss_values) + 1)
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure(figsize = (12, 9.5))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
    plt.plot(epochs,val_loss_values, 'b', label = 'Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.subplot(1,2,2)
    plt.plot(epochs, acc_values, 'go', label = 'Training acc')
    plt.plot(epochs, val_acc_values, 'g', label = 'Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    
    sys.exit("Tuning Finished, adjust parameters and re-train the model") # stop the code if still in tuning phase.

'''Fit all the data for transfer learning and train the final CNN'''
model.fit(ImageTensor, LabelTensor, batch_size=BatchSize, epochs=TrainingEpochs, verbose=1)


"""Save the model and class key"""
FullModelPath = TrainPath + ModelOutputName + '.h5'
model.save(FullModelPath)

ClassKeyName = TrainPath + ModelOutputName + '.csv'
ClassKey.to_csv(path_or_buf = ClassKeyName, index = False)





