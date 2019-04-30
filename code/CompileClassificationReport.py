# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:33:42 2018

@author: dgg0pc

Compile classification reports
this utility runs through the individual outputs of SelfSupervisedClassification and 
compiles the results into a single spreadsheet with a column structure that 
works well with Pandas row selection and Seaborn switches 
for hues and other visualisation options
"""

###############################################################################
""" Libraries"""

import numpy as np
import pandas as pd
import os.path



#############################################################
"""User data input. Fill in the info below before running"""
#############################################################

ScorePath = "Empty"
Experiment = 'Empty'
model = 'Empty'
Phase2Type = 'Empty' #indicate wether the phase 2 ML algorithm was RF or MLP

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




##################################################################
""" HELPER FUNCTIONS SECTION"""
##################################################################

##################################################################
#Save classification reports to csv with Pandas
def classification_report_csv(report, filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
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
    






##############################################################################


###############################################################################
"""Compile a single classification report from all the individual image reports"""

TestRiverTuple = (TestRiverName1, TestRiverName2, TestRiverName3, TestRiverName4, TestRiverName5,
                  TestRiverName6, TestRiverName7, TestRiverName8, TestRiverName9, TestRiverName10,
                  TestRiverName11, TestRiverName12)

#shave the empty slots off of RiverTuple
for r in range(11,0, -1):
    if 'Empty' in TestRiverTuple[r]:
        TestRiverTuple = TestRiverTuple[0:r]



#Get the CNN (phase 1) reports compiled.
TypeName = 'CNN_' 
MasterDict = {'F1':[0], 'Support':[0], 'RiverName':'Blank', 'Type':'CNN', 'Sample':'Blank', 'Class':'Blank'}
MasterDF = pd.DataFrame(MasterDict)
for f in range(0,len(TestRiverTuple)):

    print('Compiling CNN reports for ' + TestRiverTuple[f])
    for i in range(0,32000): #32000 will cover the large numbers in the StMarg set
        #if 'Kinogawa' in TestRiverTuple[f]:
         #   ReportPath = ScorePath + TypeName + TestRiverTuple[f] + format(i,'05d') + '_'  + 'Kinogawa'+'_5River'+'.csv'
        #else:
        ReportPath = ScorePath + TypeName + TestRiverTuple[f] + format(i,'05d') + '_'+ Experiment +'.csv'
        
        if os.path.exists(ReportPath):
            DF = pd.read_csv(ReportPath)
            FileValues = np.zeros((2))
            F1values = np.zeros((6,2))
            for c in range(1,6):
                if c in DF['class'].values:
                    FileValues[0] = 100*np.asarray(DF.loc[DF['class'] == c, 'f1_score'])
                    FileValues[1] = np.asarray(DF.loc[DF['class'] == c, 'support'])
                    F1values[c,0] = FileValues[0]
                    F1values[c,1] = FileValues[1]
                    if (c==1 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'InSample', 'Class':'Water', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)						
                    elif (c==2  and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'InSample', 'Class':'Sediment', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==3 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'InSample', 'Class':'Green Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
                    elif (c==4 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'InSample', 'Class':'Senesc. Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
                    elif (c==5 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'InSample', 'Class':'Paved Road', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
                    elif (c==1 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'OutOfSample', 'Class':'Water', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
                    elif (c==2  and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'OutOfSample', 'Class':'Sediment', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
                    elif (c==3 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'OutOfSample', 'Class':'Green Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
                    elif (c==4 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'OutOfSample', 'Class':'Senesc. Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
                    elif (c==5 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'OutOfSample', 'Class':'Paved Road', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF], sort = False)
            if f>5:
                WF1 = np.sum((F1values[:,0]*F1values[:,1]))/np.sum(F1values[:,1])
                FileDict = {'F1':[WF1], 'Support':[np.sum(F1values[:,1])], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'OutOfSample', 'Class':'ALL', 'Experiment':Experiment, 'model':model}
                FileDF = pd.DataFrame(FileDict)
                MasterDF = pd.concat([MasterDF, FileDF])
            else:
                WF1 = np.sum((F1values[:,0]*F1values[:,1]))/np.sum(F1values[:,1])
                FileDict = {'F1':[WF1], 'Support':[np.sum(F1values[:,1])], 'RiverName':TestRiverTuple[f], 'Type':'CNN', 'Sample':'InSample', 'Class':'ALL', 'Experiment':Experiment, 'model':model}
                FileDF = pd.DataFrame(FileDict)
                MasterDF = pd.concat([MasterDF, FileDF])

                    
            
#Get the MLP (phase 2)Empty Classification reports compiled. 
 


TypeName = Phase2Type + '_' 
for f in range(0,len(TestRiverTuple)):

    print('Compiling phase 2 reports for ' + TestRiverTuple[f])
    for i in range(0,32000): #32000 will cover the large numbers in the StMarg set

        ReportPath = ScorePath + TypeName + TestRiverTuple[f] + format(i,'05d') + '_'+ Experiment +'.csv'
        
        if os.path.exists(ReportPath):
            DF = pd.read_csv(ReportPath)
            FileValues = np.zeros((2))
            F1values = np.zeros((6,2))
            for c in range(1,6):
                if c in DF['class'].values:
                    FileValues[0] = 100*np.asarray(DF.loc[DF['class'] == c, 'f1_score'])
                    FileValues[1] = np.asarray(DF.loc[DF['class'] == c, 'support'])
                    F1values[c,0] = FileValues[0]
                    F1values[c,1] = FileValues[1]
                    if (c==1 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'InSample', 'Class':'Water', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])						
                    elif (c==2  and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'InSample', 'Class':'Sediment', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==3 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'InSample', 'Class':'Green Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==4 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'InSample', 'Class':'Senesc. Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==5 and f<6 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'InSample', 'Class':'Paved Road', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==1 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'OutOfSample', 'Class':'Water', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==2  and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'OutOfSample', 'Class':'Sediment', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==3 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'OutOfSample', 'Class':'Green Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==4 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'OutOfSample', 'Class':'Senesc. Veg.', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
                    elif (c==5 and f>5 and FileValues[1] > 0):
                        FileDict = {'F1':[FileValues[0]], 'Support':[FileValues[1]], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'OutOfSample', 'Class':'Paved Road', 'Experiment':Experiment, 'model':model}
                        FileDF = pd.DataFrame(FileDict)
                        MasterDF = pd.concat([MasterDF, FileDF])
            if f>5:
                WF1 = np.sum((F1values[:,0]*F1values[:,1]))/np.sum(F1values[:,1])
                FileDict = {'F1':[WF1], 'Support':[np.sum(F1values[:,1])], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'OutOfSample', 'Class':'ALL', 'Experiment':Experiment, 'model':model}
                FileDF = pd.DataFrame(FileDict)
                MasterDF = pd.concat([MasterDF, FileDF])
            else:
                WF1 = np.sum((F1values[:,0]*F1values[:,1]))/np.sum(F1values[:,1])
                FileDict = {'F1':[WF1], 'Support':[np.sum(F1values[:,1])], 'RiverName':TestRiverTuple[f], 'Type':Phase2Type, 'Sample':'InSample', 'Class':'ALL', 'Experiment':Experiment, 'model':model}
                FileDF = pd.DataFrame(FileDict)
                MasterDF = pd.concat([MasterDF, FileDF])
 
SaveName = ScorePath + 'Compiled_' + Experiment + '.csv'
MasterDF.to_csv(SaveName)