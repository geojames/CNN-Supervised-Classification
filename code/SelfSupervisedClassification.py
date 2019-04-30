# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:33:42 2018

@author: Patrice Carbonneau

Performs Self-Supervised Image CLassification with a pre-trained Convolutional Neural Network model.
User options are in the first section of code.
"""

###############################################################################
""" Libraries"""
from keras import regularizers
from keras import optimizers
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from skimage import io
import skimage.transform as T
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import median, entropy, modal
import os.path
from sklearn import metrics
from skimage.morphology import disk
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import copy
import sys
from IPython import get_ipython #this can be removed if not using Spyder




#############################################################
"""User data input. Fill in the info below before running"""
#############################################################

ModelPath = ""  #should be the training path from previous run of TrainCNN_
PredictPath = ""
TestRiverName1 = "Empty"  #
TestRiverName2 = "Empty"  # 
TestRiverName3 = "Empty"  # 
TestRiverName4 = "Empty"
TestRiverName5 = "Empty"
TestRiverName6 = 'Empty'
TestRiverName7 = "Empty"
TestRiverName8 = "Empty"
TestRiverName9 = "Empty"
TestRiverName10 = "Empty"
TestRiverName11 = "Empty"
TestRiverName12 = "Empty"
ModelName = ''   #The retrained convnet to use, do not specify extension
ScorePath = ""
ExperimentName = '' #ID to append to output performance files


UseSmote = False#Turn SMOTE-ENN resampling on and off
size = 50 # Size of the tiles.  This MUST be the same as the tile size used to retrain the model
MinTiles = 0 #The minimum number of contiguous tiles of size 'size' to consider as a significnat element in the image.  
NClasses = 5  #The number of classes in the data. This MUST be the same as the classes used to retrain the model
RecogThresh = 0 #minimum prob of the top-1 predictions to keep
FirstImage = 0
BiggestImage = 9999 #Enter the number, can be approximate but bigger, of the last image
Ndims = 4 # Feature Dimensions. 4 if using entropy in the self-supervised classification, 3 if just RGB
DropRate = 0.5

#Set MLP below to True to use the output of the CNN as training data in an MLP, defined below. False will use a random forest.
MLP = True 
#Choose MLP model
ModelChoice = 2 # 2 for deep model and 3 for very deep model 
LearningRate = 0.001
#If MLP is True, enter the right number of epochs. 
TrainingEpochs = 70


Chatty = 1 # set the verbosity of the model training.  Use 1 at first, 0 when confident that model is well tuned
SubSample = 0.1 #0-1 percentage of the image pixels to use in fitting the self-supervised models
MinSample = 250000 #minimum sample size
SmallestElement = 10 # Despeckle the classification to the smallest length in pixels of element remaining, just enter linear units (e.g. 3 for 3X3 pixels)

#if true this will plot a results figure for each holdout image.  Set to false if you have more than ca. 15-20 holdout images
# if false it will just plot the last one.
#in both cases, a png version will be saved to disk
DisplayHoldout =  False
OutDPI = 300 #Recommended 300 for inspection 1200 for papers, at 1200 this results in 15Mb per output figure.  

#####################################################################################################################


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
    #It then extracts the image tiles that have a classification value in the labels
    LabelVector = np.zeros(ClassTile.shape[0])
    
    for v in range(0,ClassTile.shape[0]):
        Tile = ClassTile[v,:,:,0]
        vals, counts = np.unique(Tile, return_counts = True)
        if (vals[0] == 0) and (counts[0] > 0.1 * size**2):
            LabelVector[v] = 0
        elif counts[np.argmax(counts)] >= 0.9 * size**2:
            LabelVector[v] = vals[np.argmax(counts)] 
    
    LabelVector = LabelVector[LabelVector > 0]
    ClassifiedTiles = np.zeros((np.count_nonzero(LabelVector), size,size,3))
    C = 0
    for t in range(0,np.count_nonzero(LabelVector)):
        if LabelVector[t] > 0:
            ClassifiedTiles[C,:,:,:] = ImageTile[t,:,:,:]
            C += 1
    return LabelVector, ClassifiedTiles


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
            #TileTensor[B,:,:,:] = im[y1:y2,x1:x2].reshape(size,size,d)
            TileImage[y1:y2,x1:x2] = np.argmax(PredictedTiles[B,:])
            B+=1

    return TileImage

# This is a helper function to repeat a filter on 3 colour bands.  Avoids an extra loop in the big loops below
def ColourFilter(Image):
    med = np.zeros(np.shape(Image))
    for b in range (0,3):
        img = Image[:,:,b]
        med[:,:,b] = median(img, disk(5))
    return med
 

##################################################################
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


###############################################################################
# Return a class prediction to the 1-Nclasses hierarchical classes
def SimplifyClass(ClassImage, ClassKey):
    Iclasses = np.unique(ClassImage)
    for c in range(0, len(Iclasses)):
        KeyIndex = ClassKey.loc[ClassKey['LocalClass'] == Iclasses[c]]
        Hclass = KeyIndex.iloc[0]['HierarchClass']
        ClassImage[ClassImage == Iclasses[c]] = Hclass
    return ClassImage



##########################################
#fetches the overall avg F1 score from a classification report
def GetF1(report):
    lines = report.split('\n')
    for line in lines[0:-1]:
        if 'weighted' in line:
            dat = line.split(' ')
    
    return dat[17]

##############################################################################
"""Instantiate Random Forest and Dense Neural Network pixel-based classifiers""" 
   
#Instantiate the Random Forest estimator
EstimatorRF = RFC(n_estimators = 150, n_jobs = 8, verbose = Chatty) #adjust this to your processors




# define deep the model with L2 regularization and dropout
def deep_model_L2D():
	# create model
    model = Sequential()
    model.add(Dense(256, kernel_regularizer= regularizers.l2(0.001), input_dim=Ndims, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(NClasses, kernel_initializer='normal', activation='softmax'))
    
    #Tune an optimiser
    Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    return model

# define the very deep model with L2 regularization and dropout
def very_deep_model_L2D():
	# create model
    model = Sequential()
    model.add(Dense(512, kernel_regularizer= regularizers.l2(0.001), input_dim=Ndims, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(DropRate))
    model.add(Dense(128, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(DropRate))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(DropRate))
    model.add(Dense(NClasses, kernel_initializer='normal', activation='softmax'))
    
    #Tune an optimiser
    Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    return model



# Instantiate and fit a MLP model estimator from choices above
if ModelChoice == 1:
    print("Don't waste your time in shallow learning...")
    
elif ModelChoice == 2:
    EstimatorNN = KerasClassifier(build_fn=deep_model_L2D, epochs=TrainingEpochs, batch_size=250000, verbose=Chatty)
    
elif ModelChoice == 3:
    EstimatorNN = KerasClassifier(build_fn=very_deep_model_L2D, epochs=TrainingEpochs, batch_size=250000, verbose=Chatty)
    
else:
    sys.exit("Invalid Model Choice")
    



###############################################################################
"""Load the convnet model"""
#print('Loading re-trained convnet model produced by a run ThirdTimeLucky_(CNN model name).py')
print('Loading ' + ModelName + '.h5')
FullModelPath = ModelPath + ModelName + '.h5'
ConvNetmodel = load_model(FullModelPath)

ClassKeyPath = ModelPath + ModelName + '.csv'
ClassKey = pd.read_csv(ClassKeyPath)

###############################################################################
"""Classify the holdout images with Self-Supervised Classification"""

TestRiverTuple = (TestRiverName1, TestRiverName2, TestRiverName3, TestRiverName4, TestRiverName5,
                  TestRiverName6, TestRiverName7, TestRiverName8, TestRiverName9, TestRiverName10,
                  TestRiverName11, TestRiverName12)

#shave the empty slots off of RiverTuple
for r in range(11,0, -1):
    if 'Empty' in TestRiverTuple[r]:
        TestRiverTuple = TestRiverTuple[0:r]

for f in range(0,len(TestRiverTuple)):
  
    for i in range(FirstImage,BiggestImage): #Image numbers up to 99999. But excess figures will crash Spyder if you actually have 1000
        ImagePath = PredictPath + TestRiverTuple[f] + format(i,'05d') +'.jpg'
        ClassPath = PredictPath + 'SCLS_' + TestRiverTuple[f] + format(i,'05d')  +'.tif' #watch image format types
        if os.path.exists(ImagePath):
            print('Self-supervised classification of ' + ImagePath)
            Im3D = io.imread(ImagePath)
            #print(isinstance(Im3D,uint8))
            if len(Im3D) == 2:
                Im3D = Im3D[0]
            Class = io.imread(ClassPath, as_gray=True)
            if (Class.shape[0] != Im3D.shape[0]) or (Class.shape[1] != Im3D.shape[1]):
                print('WARNING: inconsistent image and class mask sizes for ' + ImagePath)
                Class = T.resize(Class, (Im3D.shape[0], Im3D.shape[1]), preserve_range = True) #bug handling for vector
            ClassIm = copy.deepcopy(Class)
            #Tile the images to run the convnet
            ImCrop = CropToTile (Im3D, size)
            I_tiles = split_image_to_tiles(ImCrop, size)
            I_tiles = np.int16(I_tiles) / 255
            #Apply the convnet
            print('Detecting self-supervised training areas')
            PredictedTiles = ConvNetmodel.predict(I_tiles, batch_size = 32, verbose = Chatty)
            #Convert the convnet one-hot predictions to a new class label image
            PredictedTiles[PredictedTiles < RecogThresh] = 0
            PredictedClass = class_prediction_to_image(Class, PredictedTiles, size)
            PredictedClass = SimplifyClass(PredictedClass, ClassKey)
            #Set classes to 0 if they do not have MinTiles detected by the CNN
            
            #for c in range(0,NClasses+1):
            #    count = np.sum(PredictedClass.reshape(-1,1) == c)
            #    if count <= MinTiles*size*size:
            #        PredictedClass[PredictedClass == c] = 0
            if MinTiles > 0:
                PredictedClass = modal(np.uint8(PredictedClass), np.ones((2*MinTiles*size+1,2*MinTiles*size+1)))
                #PredictedClass = modal(np.uint8(PredictedClass), disk(2*(MinTiles*size*size)+1))
								
            #Prep the pixel data from the cropped image and new class label image
            print('Processing Entropy and Median filter')
            Entropy = entropy(Im3D[:,:,0], selem = np.ones([11,11]), shift_x = 3,  shift_y = 0)
            MedImage = ColourFilter(Im3D) #Median filter on all 3 bands
            rv = MedImage[:,:,0].reshape(-1,1)#Split and vectorise the bands
            gv = MedImage[:,:,1].reshape(-1,1)
            bv = MedImage[:,:,2].reshape(-1,1)
            #Vectorise the bands, use the classification prdicted by the CNN
            #m = np.ndarray.flatten(PredictedClass).reshape(-1,1) 
            #rv = np.ndarray.flatten(r).reshape(-1,1)
            #gv = np.ndarray.flatten(g).reshape(-1,1)
            #bv = np.ndarray.flatten(b).reshape(-1,1)
            #Entropyv = np.ndarray.flatten(Entropy).reshape(-1,1)
            m = PredictedClass.reshape(-1,1) 
            Entropyv = Entropy.reshape(-1,1)
            

            ColumnDat = np.concatenate((rv,gv,bv,Entropyv,m), axis = 1)
            #Rescale the data for the fitting work
            SCAL = StandardScaler()
            ScaledValues = SCAL.fit_transform(ColumnDat[:,0:-1])
            ColumnDat[:,0:-1] = ScaledValues
            #Eliminate the zeros in the mask from minimum tiles
            ColumnDat = ColumnDat[ColumnDat[:,4]!=0]
            
            m=ColumnDat[:,-1]
            #Build the predictor from the CNN classified mask
            #Subsample the pixels
            sample_size = np.int64(SubSample * ColumnDat.shape[0])
            if (sample_size < MinSample) and (ColumnDat.shape[0] > MinSample):
                sample_size = MinSample
            elif (sample_size < MinSample) and (ColumnDat.shape[0] < MinSample):
                sample_size = ColumnDat.shape[0]
                print('WARNING: small sample size for predictor fit')
            idx = np.random.randint(low = 1, high = int(len(ColumnDat)-1), size=sample_size) #using numpy so should be fast
            ColumnDat = ColumnDat[idx,:]
            X_presmote = ColumnDat[:,0:4] 
            Y_presmote = ColumnDat[:,4]
            if UseSmote and len(np.unique(Y_presmote))>1: #SMOTE doesn't work with 1 class 
                print('Correcting class imbalance with SMOTE-ENN')
                smote_enn = SMOTEENN(sampling_strategy ='auto', random_state=0)
                X, Y = smote_enn.fit_resample(X_presmote, Y_presmote)
            else:
                print('not using SMOTE methods')
                X = X_presmote
                Y = Y_presmote
                
                    
             

            
            if MLP:
                print('Fitting MLP Classifier on ' + str(len(X)) + ' pixels')
                EstimatorNN.fit(X, Y)
            else:
             
                print('Fitting Random Forest Classifier on ' + str(len(X)) + ' pixels')
                EstimatorRF.fit(X, Y)
                
            #Fit the predictor to all pixels
            FullDat = np.concatenate((rv,gv,bv,Entropyv), axis = 1)
            del(rv,gv,bv,Entropyv, MedImage)
            SCAL = StandardScaler()
            ScaledValues = SCAL.fit_transform(FullDat)
            if MLP:
                PredictedPixels = EstimatorNN.predict(ScaledValues)
            else:
                PredictedPixels = EstimatorRF.predict(ScaledValues)
            
            #Reshape the predictions to image format and display
            PredictedImage = PredictedPixels.reshape(Entropy.shape[0], Entropy.shape[1])
            if SmallestElement > 0:
                PredictedImage = modal(np.uint8(PredictedImage), disk(2*SmallestElement+1)) #clean up the class with a mode filter


            #Produce classification reports 
            Class = Class.reshape(-1,1)
            PredictedImageVECT = PredictedImage.reshape(-1,1) #This is the pixel-based prediction
            PredictedClassVECT = PredictedClass.reshape(-1,1) # This is the CNN tiles prediction
            PredictedImageVECT = PredictedImageVECT[Class != 0]
            PredictedClassVECT = PredictedClassVECT[Class != 0]
            Class = Class[Class != 0]
            Class = np.int32(Class)
            PredictedImageVECT = np.int32(PredictedImageVECT)
            reportSSC = metrics.classification_report(Class, PredictedImageVECT, digits = 3)
            reportCNN = metrics.classification_report(Class, PredictedClassVECT, digits = 3)
            print('CNN tiled classification results for ' + ImagePath)
            print(reportCNN)
            print('\n')
            print('Self-Supervised classification results for ' + ImagePath)
            print(reportSSC)
            #print('Confusion Matrix:')
            #print(metrics.confusion_matrix(Class, PredictedImageVECT))
            print('\n')
            if MLP:
                SSCname = ScorePath + 'SSC_MLP_' + TestRiverTuple[f] + format(i,'05d') +  '_' + ExperimentName + '.csv'    
                classification_report_csv(reportSSC, SSCname)
            else:
                SSCname = ScorePath + 'SSC_RF_' + TestRiverTuple[f] + format(i,'05d') +  '_' + ExperimentName + '.csv'    
                classification_report_csv(reportSSC, SSCname)
            CNNname = ScorePath + 'CNN_' + TestRiverTuple[f] + format(i,'05d') +  '_' + ExperimentName + '.csv'    
            classification_report_csv(reportCNN, CNNname)            
            
            #Display results
            #PredictedImage = PredictedPixels.reshape(Entropy.shape[0], Entropy.shape[1])
            for c in range(0,6): #this sets 1 pixel to each class to standardise colour display
                ClassIm[c,0] = c
                PredictedClass[c,0] = c
                PredictedImage[c,0] = c
            #get_ipython().run_line_magic('matplotlib', 'qt')
            plt.figure(figsize = (12, 9.5)) #reduce these values if you have a small screen
            plt.subplot(2,2,1)
            plt.imshow(Im3D)
            plt.title('Classification results for ' + TestRiverTuple[f] + format(i,'05d') +'.jpg', fontweight='bold')
            plt.xlabel('Input RGB Image', fontweight='bold')
            plt.subplot(2,2,2)
            cmapCHM = colors.ListedColormap(['black','lightblue','orange','green','yellow','red'])
            plt.imshow(ClassIm, cmap=cmapCHM)
            plt.xlabel('Validation Labels', fontweight='bold')
            class0_box = mpatches.Patch(color='black', label='Unclassified')
            class1_box = mpatches.Patch(color='lightblue', label='Water')
            class2_box = mpatches.Patch(color='orange', label='Sediment')
            class3_box = mpatches.Patch(color='green', label='Green Veg.')
            class4_box = mpatches.Patch(color='yellow', label='Senesc. Veg.')
            class5_box = mpatches.Patch(color='red', label='Paved Road')
            ax=plt.gca()
            ax.legend(handles=[class0_box, class1_box,class2_box,class3_box,class4_box,class5_box])
            plt.subplot(2,2,3)
            plt.imshow(PredictedClass, cmap=cmapCHM)
            plt.xlabel('CNN tiles Classification. F1: ' + GetF1(reportCNN), fontweight='bold')
            plt.subplot(2,2,4)
            cmapCHM = colors.ListedColormap(['black', 'lightblue','orange','green','yellow','red'])
            plt.imshow(PredictedImage, cmap=cmapCHM)
            if MLP:
                plt.xlabel('Self-Supervised Classification (MLP). F1: ' + GetF1(reportSSC), fontweight='bold' )
            else:
                plt.xlabel('Self-Supervised Classification (RF). F1: ' + GetF1(reportSSC), fontweight='bold' )
            FigName = ScorePath + 'SSC_' + 'OutFig_' + TestRiverTuple[f] + format(i,'05d') +'.png'
            plt.savefig(FigName, dpi=OutDPI)
            if not DisplayHoldout:
                plt.close()
            





            
            
#Write out a classified images with a duplicated worldfile
'''This section needs to be written. At the moment, there is no proper output aside from figures and the overall report as a csv'''


