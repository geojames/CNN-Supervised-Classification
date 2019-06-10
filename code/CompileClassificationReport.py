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
Name:           CompileClassificationReport.py
Compatibility:  Python 3.6
Description:    this utility runs through the individual outputs of 
                SelfSupervisedClassification and compiles the results into a
                single spreadsheet with a column structure that works well with
                Pandas row selection and Seaborn switches for hues and other
                visualisation options
Requires:       numpy, pandas, glob

Dev Revisions:  JTD - 19/6/10 - Updated and optimied for Pandas
                    
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

###############################################################################
""" Libraries"""

import numpy as np
import pandas as pd
import glob
import os.path


#############################################################
"""User data input. Fill in the info below before running"""
#############################################################

ScorePath = "Empty"     #output folder
Experiment = 'Empty'
model = 'Empty'         #this will be a label, not the exact model name
Phase2Type = 'MLP'      #indicate wether the phase 2 ML algorithm was MLP (default) or RF

# Path checks- checks for folder ending slash, adds if nessesary

if ('/' or "'\'") not in ScorePath[-1]:
    ScorePath = ScorePath +'/'

###############################################################################
"""Compile a single classification report from all the individual image reports"""

# Getting River Names from the input files
# Glob list fo all jpg images, get unique names form the total list
cnn_csv = glob.glob(ScorePath+"CNN_*.csv")
TestRiverTuple = []
for c in cnn_csv:
    TestRiverTuple.append(os.path.basename(c).split('_')[1])
TestRiverTuple = np.unique(TestRiverTuple)

# Get the CNN (phase 1) reports compiled

#establish a DF for the CNN results
CNN_df = pd.DataFrame(np.zeros([1,8]),columns = ['class','f1_score','support','RiverName','Image','type','experiment','model'])
                    
# for each river, get all the csv files that match river name
#   for each csv file, extract: class, f1 and support scores to new DF
#       Clean and concat the CNN csv files as we go
#   Renames classes and add type, exp, model column info
for f,riv in enumerate(TestRiverTuple):
    print('Compiling CNN reports for ' + riv)
    cnn_csv = glob.glob(ScorePath+"CNN_" + riv + "*.csv")
    
    for i,csv in enumerate(cnn_csv):
        DF = pd.read_csv(csv)                       
        FileDf = DF.filter(['class','f1_score','support'],axis=1)
        FileDf = FileDf.drop(FileDf[(FileDf.f1_score == 0) & (FileDf.support == 0)].index)
        FileDf = FileDf.append(pd.DataFrame([['ALL',np.sum((FileDf.f1_score*FileDf.support))/np.sum(FileDf.support),np.sum(FileDf.support)]],columns=['class','f1_score','support']))
        FileDf['RiverName'] = riv
        FileDf['Image'] = os.path.basename(csv).split('_')[1] + "_" + os.path.basename(csv).split('_')[2]
        CNN_df = pd.concat([CNN_df, FileDf], sort = False)
        
    CNN_df = CNN_df[1:]   
    CNN_df['f1_score'] = 100 * CNN_df['f1_score']
    CNN_df['class'][CNN_df['class']==1] = 'Water'
    CNN_df['class'][CNN_df['class']==2] = 'Sediment'
    CNN_df['class'][CNN_df['class']==3] = 'Green Veg.'
    CNN_df['class'][CNN_df['class']==4] = 'Senesc. Veg.'
    CNN_df['class'][CNN_df['class']==5] = 'Paved Road'
    CNN_df['type'] = 'CNN'
    CNN_df['experiment'] = Experiment
    CNN_df['model'] = model

# Get the MLP (phase 2) classification reports compiled. 

# establish Phase 2 DF  
p2_df = pd.DataFrame(np.zeros([1,8]),columns = ['class','f1_score','support','RiverName','Image','type','experiment','model'])])
                  
# for each river, get all the csv files that match river name
#   for each csv file, extract: class, f1 and support scores to new DF
#       Clean and concat the CNN csv files as we go
#   Renames classes and add type, exp, model column info
for f,riv in enumerate(TestRiverTuple):
    print('Compiling Phase 2 reports for ' + riv)
    
    if Phase2Type == 'MLP':
        p2_csv = glob.glob(ScorePath+"MLP_" + riv + "*.csv")
    elif Phase2Type == 'RF':
        p2_csv = glob.glob(ScorePath+"RF_" + riv + "*.csv")

    for i,csv in enumerate(p2_csv):

        DF = pd.read_csv(csv)                       
        FileDf = DF.filter(['class','f1_score','support'],axis=1)
        FileDf = FileDf.drop(FileDf[(FileDf.f1_score == 0) & (FileDf.support == 0)].index)
        FileDf = FileDf.append(pd.DataFrame([['ALL',np.sum((FileDf.f1_score*FileDf.support))/np.sum(FileDf.support),np.sum(FileDf.support)]],columns=['class','f1_score','support']))
        FileDf['RiverName'] = riv
        FileDf['Image'] = os.path.basename(csv).split('_')[1] + "_" + os.path.basename(csv).split('_')[2]
        p2_df = pd.concat([p2_df, FileDf], sort = False)
        
    p2_df = p2_df[1:]
    p2_df['f1_score'] = 100 * p2_df['f1_score']
    p2_df['class'][p2_df['class']==1] = 'Water'
    p2_df['class'][p2_df['class']==2] = 'Sediment'
    p2_df['class'][p2_df['class']==3] = 'Green Veg.'
    p2_df['class'][p2_df['class']==4] = 'Senesc. Veg.'
    p2_df['class'][p2_df['class']==5] = 'Paved Road'
    p2_df['experiment'] = Experiment
    p2_df['model'] = model
    
    if Phase2Type == 'MLP':
        p2_df['type'] = 'MLP' 
    elif Phase2Type == 'RF':
        p2_df['type'] = 'RF' 

SaveName = ScorePath + 'Compiled_' + Experiment + '.csv'
pd.concat([CNN_df,p2_df]).to_csv(SaveName,index=False)
