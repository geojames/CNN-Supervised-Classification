# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'
__date__ = '26 SEPT 2019'
__version__ = '1.2'
__status__ = "stable release"
__url__ = "https://github.com/geojames/Self-Supervised-Classification"


"""
Name:           CnnSupervisedClassification_PyQGIS.py

Compatibility:  Python 3.6

Description:    Performs Self-Supervised Image CLassification with a 
                pre-trained Convolutional Neural Network model.
                User options are in the first section of code.
                
Requires:       keras, numpy, pandas, matplotlib, skimage, sklearn, gdal

Dev Revisions:  PC -  First version

                
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

# ------------------------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------------------------
""" Libraries"""
import os.path
import datetime
import time
import sys
import numpy as np
import pandas as pd
from keras import regularizers
from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from skimage import io
from skimage.filters.rank import median
from skimage.morphology import disk
from sklearn import metrics
from sklearn.utils import class_weight
import gdal, osr
from qgis.utils import iface


# ------------------------------------------------------------------------------------------------
# USER INPUT
"""User data input. Fill in the info below before running"""
# ------------------------------------------------------------------------------------------------

ModelName = 'Empty'     #should be the model name from previous run of TrainCNN.py (NO FILE ENDING)
ModelPath = 'Empty'  # path to the model
ImageFolder = 'Empty'
ImageFile = 'Empty'
ClassFile = 'Empty'#leave empty if none.  In that case no validation will be performed.
NoData = 0 #no data values if mosaic has no alpha layer

'''BASIC PARAMETER CHOICES'''
TrainingEpochs = 15 #This requires minimal experimentation to tune
Ndims = 3 # Feature Dimensions. should be 3 for RGB
NClasses = 5  #The number of classes in the data. This MUST be the same as the classes used to retrain the model
  

'''MODEL PARAMETERS''' #These would usually not be edited
DropRate = 0.5
LearningRate = 0.005



# timer
start_time = datetime.datetime.now()
loop_time = 0


# END USER INPUT
# ------------------------------------------------------------------------------------------------


    
# ------------------------------------------------------------------------------------------------
""" HELPER FUNCTIONS SECTION"""
# ------------------------------------------------------------------------------------------------
# Helper function to crop images to have an integer number of tiles. No padding is used.
def CropToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]
#END - CropToTile   
    
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
# END - split_image_to_tiles
    


# Takes the predicted tiles and reconstructs an image from them
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
            #TileTensor[B,:,:,:] = im[y1:y2,x1:x2].reshape(size,size,d)
            #TileImage[y1:y2,x1:x2] = np.nanmax(PredictedTiles[B,:])
            TileImage[y1:y2,x1:x2] = np.argmax(PredictedTiles[B,:])
            B+=1

    return TileImage
# END - class_prediction_to_image

# This is a helper function to repeat a filter on 3 colour bands.
#   Avoids an extra loop in the big loops below
def ColourFilter(Image):
    med = np.zeros(np.shape(Image))
    for b in range (0,3):
        img = Image[:,:,b]
        med[:,:,b] = median(img, disk(5))
    return med
# END - ColourFilter

#Save classification reports to csv with Pandas
def classification_report_csv(report, filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_data = line.split(' ') 
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False) 
# END - classification_report_csv

# Return a class prediction to the 1-Nclasses hierarchical classes
def SimplifyClass(ClassImage, ClassKey):
    Iclasses = np.unique(ClassImage)
    for c in range(0, len(Iclasses)):
        KeyIndex = ClassKey.loc[ClassKey['LocalClass'] == Iclasses[c]]
        Hclass = KeyIndex.iloc[0]['HierarchClass']
        ClassImage[ClassImage == Iclasses[c]] = Hclass
    return ClassImage
# END - SimplifyClass

#fetches the overall avg F1 score from a classification report
def GetF1(report):
    lines = report.split('\n')
    for line in lines[0:-1]:
        if 'weighted' in line:
            dat = line.split(' ')
    
    return dat[17]
# END - GetF1

# ------------------------------------------------------------------------------------------------
"""Instantiate MLP pixel-based classifiers""" 
# ------------------------------------------------------------------------------------------------


MLP = True
def best_model_L2D():
	# create model
    model = Sequential()
    model.add(Dense(512, kernel_regularizer= regularizers.l2(0.001), input_dim=Ndims, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))

    model.add(Dense(NClasses, kernel_initializer='normal', activation='softmax'))
    
    #Tune an optimiser
    Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    return model
    
EstimatorNN = KerasClassifier(build_fn=best_model_L2D, epochs=TrainingEpochs, 
                                  batch_size=250000, verbose=0)
    
# ------------------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------------------
    

if not QgsProject.instance().mapLayers().values():
    print('ERROR: this script assumes that the image to be classified is open in QGIS with project CRS correctly set')
else:
    '''Form the file names'''
    ImageName = os.path.join(ImageFolder, ImageFile)
    ClassName = os.path.join(ImageFolder, ClassFile)
    OutputFile1 = 'CLASS_CNN_'+ImageFile
    OutputFile2 = 'CLASS_CSC_'+ImageFile
    OutputName1 =   os.path.join(ImageFolder, OutputFile1) #Location of the CNN class with fullpath
    OutputName2 =   os.path.join(ImageFolder, OutputFile2) #Location of the MLP class with fullpath
    Chatty = 0

    """Load the convnet model"""
    #print('Loading re-trained convnet model produced by a run of TrainCNN.py')
    print('Loading ' + ModelName + '.h5')
    time.sleep(1)
    FullModelPath = os.path.join(ModelPath, ModelName + '.h5')
    ConvNetmodel = load_model(FullModelPath)
    ClassKeyPath = os.path.join(ModelPath, ModelName + '.csv')
    ClassKey = pd.read_csv(ClassKeyPath)


    """Classify the holdout images with CNN-Supervised Classification"""
    size = 50 #Do not edit. The base models supplied all assume a tile size of 50.

        # timer
    start_time = datetime.datetime.now()
    loop_time = 0


    print('CNN-supervised classification of ' + ImageName)
    time.sleep(1)   
    Im3D = io.imread(ImageName)
    if Im3D.shape[2]>3:
        ALPHA = Im3D[:,:,3]
        Im3D = Im3D[:,:,0:3]
    else:
        ALPHA = ~(((Im3D[:,:,0]==NoData) & (Im3D[:,:,1]==NoData) & (Im3D[:,:,2]==NoData)))

            
              

                
    #Tile the images to run the convnet
    ImCrop = CropToTile (Im3D, size)
    I_tiles = split_image_to_tiles(ImCrop, size)
    I_tiles = np.int16(I_tiles) / 255
    del(Im3D, ImCrop)
    #Apply the convnet
    print('Detecting CNN-supervised training areas')
    time.sleep(1)
    PredictedTiles = ConvNetmodel.predict(I_tiles, batch_size = 64, verbose = Chatty)
    del(I_tiles)
    #Convert the convnet one-hot predictions to a new class label image
    PredictedClass = class_prediction_to_image(ALPHA, PredictedTiles, size)
    del(PredictedTiles, ConvNetmodel)
    PredictedClass = SimplifyClass(PredictedClass, ClassKey)
    PredictedClass[ALPHA==0]=0 #Set classes to 0 for off-image patches (transparent values in the alpha)
    '''Georeferenced Export of CNN class'''
        
    ImageFile = gdal.Open(ImageName)
    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(OutputName1, PredictedClass.shape[1], PredictedClass.shape[0], gdal.GDT_Byte)
    outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
    outRasterSRS = osr.SpatialReference()
    project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
    project_crs = int(project_crs_name[5:])
    outRasterSRS.ImportFromEPSG(project_crs)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outRaster.GetRasterBand(1).WriteArray(PredictedClass)
    outRaster.FlushCache()  # saves image to disk
    outRaster = None        # close output file
    ImageFile = None        # close input file
        ##Open the new Class Raster data in QGIS
    print('Displaying Classification')
    time.sleep(1)
    fileInfo = QFileInfo(OutputName1)
    baseName = fileInfo.baseName()
    rlayer = QgsRasterLayer(OutputName1, baseName)
    if not rlayer.isValid():
        print ('Layer failed to load!')
    else:
        QgsProject.instance().addMapLayer(rlayer)

    '''MLP part'''
    Im3D = io.imread(ImageName)
    Im3D = Im3D[:,:,0:3]
    print('3x3 Median filter' )
    time.sleep(1)
    MedImage = ColourFilter(Im3D) #Median filter on all 3 bands
    del(Im3D)
    print('Preparing data for MLP')
    time.sleep(1)
    rv = MedImage[:,:,0].reshape(-1,1) / 255#Split and vectorise the bands
    gv = MedImage[:,:,1].reshape(-1,1) / 255
    bv = MedImage[:,:,2].reshape(-1,1) / 255
    del(MedImage)
    PredictedClass = PredictedClass.reshape(-1,1)


    ColumnDat = np.concatenate((rv,gv,bv,PredictedClass), axis = 1)
    ColumnDat = ColumnDat[ColumnDat[:,-1]!=0]#ignore class 0
    del(PredictedClass)
    X= ColumnDat[:,0:-1]
    Y=ColumnDat[:,-1]
    del(ColumnDat)
    #Adjust class weights
    weights = class_weight.compute_class_weight('balanced', np.unique(Y),Y)

    #Fit the training
    EstimatorNN.fit(X, Y, class_weight=weights)
    del(X,Y)
    #Estiange for all pixels in the mosaic
    FullDat = np.concatenate((rv,gv,bv), axis = 1)
    del(rv,gv,bv)
    PredictedPixels = EstimatorNN.predict(FullDat)
    del(FullDat)
    PredictedImage = PredictedPixels.reshape(ALPHA.shape[0], ALPHA.shape[1])
    PredictedImage[ALPHA==0]=0
    del(ALPHA, EstimatorNN, PredictedPixels)
        #plt.imshow(PredictedImage)
    '''Georeferenced Export of the final CSC class'''
    ImageFile = gdal.Open(ImageName)
    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(OutputName2, PredictedImage.shape[1], PredictedImage.shape[0], gdal.GDT_Byte)
    outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
    outRasterSRS = osr.SpatialReference()
    project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
    project_crs = int(project_crs_name[5:])
    outRasterSRS.ImportFromEPSG(project_crs)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outRaster.GetRasterBand(1).WriteArray(PredictedImage)
    outRaster.FlushCache()  # saves image to disk
    outRaster = None        # close output file
    ImageFile = None        # close input file
    ##Open the new Class Raster data in QGIS
    print('Displaying CSC Classification')
    time.sleep(1)
    fileInfo = QFileInfo(OutputName2)
    baseName = fileInfo.baseName()
    rlayer = QgsRasterLayer(OutputName2, baseName)
    if not rlayer.isValid():
        print ('Layer failed to load!')
    else:
            QgsProject.instance().addMapLayer(rlayer)



    '''Quality report if validation available'''

    try:
        Class = io.imread(ClassName)
        #Produce classification reports 
        Class = Class.reshape(-1,1)
        PredictedImageVECT = PredictedImage.reshape(-1,1) #This is the pixel-based prediction
        PredictedImageVECT = PredictedImageVECT[Class != 0]
        Class = Class[Class != 0]
        Class = np.int32(Class)
        PredictedImageVECT = np.int32(PredictedImageVECT)
        reportSSC = metrics.classification_report(Class, PredictedImageVECT, digits = 3)


        print('CNN-Supervised classification results for ' + ImageName)
        print(reportSSC)
        print('Confusion Matrix:')
        print(metrics.confusion_matrix(Class, PredictedImageVECT))
        print('\n')


            
            
    except:
        print('no validation data found')

        #
        #
            
    print("*-*-*-*-*")
    print("|")
    print("|   Total Classification Time =",datetime.datetime.now() - start_time)
    print("|")
    print("*-*-*-*-*")
