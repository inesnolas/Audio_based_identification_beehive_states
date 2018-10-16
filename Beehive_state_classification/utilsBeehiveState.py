
import glob
import os
import librosa
import pdb
import csv
import json
import re
import numpy as np
import random
import librosa.display
import IPython.display as ipd
from sklearn import preprocessing
import sys
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
from collections import Counter
from matplotlib import pyplot as plt

sys.path.append('C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\Audio_based_indentification_beehive_State\\Bee_NotBee_classification')
from info import i, printb, printr, printp, print



def change_Ancona_audioFiles_Names(data_path):
#we want to have the state of the hive in the name of each file, but it only comes in the name of the folder.

    foldersNames_list=[]
    content = os.listdir(data_path)
    
    for cont in content:
        if os.path.isdir(data_path+cont):
            foldersNames_list.append(cont)

    for folder_name in foldersNames_list:
        audiofilenames_list =  os.listdir(data_path + folder_name)
        for audioFile_name in audiofilenames_list:
            os.rename(data_path + folder_name+ os.sep+ audioFile_name, data_path + folder_name+ os.sep+folder_name+'_'+ audioFile_name)
    return
    
    
# def report_SVM_beehiveState_results(report_fileName, clf, y_pred_proba_train, y_pred_train, y_pred_proba_test, y_pred_test, X_train, GT_y_train, X_test, GT_y_test):
# ## saves results to file and prints to screen;

def SVM_Classification_BeehiveSTATE(X_flat_train, y_train, X_flat_test, y_test, kerneloption='rbf'):

    print('\n')
    printb('Starting classification with SVM:')
    Test_Preds=[]
    Train_Preds=[]
    Test_Preds_Proba=[]
    Train_Preds_Proba=[]
    Test_GroundT=[]
    Train_GroundT=[]
   
    print('\n')
    printb('classification Beehive State into : Active or Missing Queen')
        
    #train :
    CLF = svm.SVC(kernel=kerneloption, probability=True)
    CLF.fit(X_flat_train, y_train)
    y_pred_train = CLF.predict(X_flat_train)
    y_pred_proba_train = CLF.predict_proba(X_flat_train)
    
    Train_GroundT = y_train
    Train_Preds = y_pred_train
    Train_Preds_Proba = y_pred_proba_train[:,1]
    
    # test:
    y_pred_test = CLF.predict(X_flat_test)
    y_pred_proba_test = CLF.predict_proba(X_flat_test)
    Test_GroundT= y_test
    Test_Preds = y_pred_test
    Test_Preds_Proba = y_pred_proba_test[:,1]

    return CLF, Test_GroundT, Train_GroundT, Test_Preds, Train_Preds, Test_Preds_Proba, Train_Preds_Proba 
    
    
    
def report_SVM_beehiveState_results(summary_filename, path_results, thresholds, CLF, Test_GroundT, Train_GroundT, Test_Preds, Train_Preds, Test_Preds_Proba, Train_Preds_Proba, classification_idSTRING, testFilenames,  chunk_size, save='yes'):

    

    if not os.path.isfile(summary_filename):
        with open(summary_filename, 'w') as csvfile:
            wtr = csv.writer(csvfile, delimiter=',')
            wtr.writerow(['ExperienceParameters', 'AccuracyTRAIN', 'AUC_TRAIN', 'gtACTIVEpACTIVE_TRAIN', 'gtMQUEENpACTIVE_TRAIN', 'gtACTIVEpMQUEEN_TRAIN', 'gtMQUEENpMQUEEN_TRAIN','ShannonEnthropy_TRAIN','AccuracyTEST', 'AUC_TEST','ConfusionMatrixTEST_gtACTIVEpACTIVE_gtMQUEENpACTIVE_gtACTIVEpMQUEEN_gtMQUEENpMQUEEN', 'PrecisionTEST_on_MQUEEN', 'RecallTEST_on_MQUEEN', 'PrecisionTEST_on_ACTIVE', 'RecallTEST_on_ACTIVE', 'gtACTIVEpACTIVE_TEST', 'gtMQUEENpACTIVE_TEST', 'gtACTIVEpMQUEEN_TEST', 'gtMQUEENpMQUEEN_TEST', 'ShannonEnthropyTEST','accuracyTEST_on_balancedDatasets'])
    
        csvfile.close()
    # transform labels into boolean type for easiness
    PRED_TEST=Test_Preds[:]
    PRED_TRAIN=Train_Preds[:]
    PRED_TEST_PROBA=Test_Preds_Proba[:]
    PRED_TRAIN_PROBA=Train_Preds_Proba[:]
    GT=Test_GroundT[:]
    GT_TRAIN=Train_GroundT[:]
   
    
    #Evaluate classifier:
            
    
    print("Classification report for classifier %s:\n%s\n"
      % (CLF, metrics.classification_report(GT, PRED_TEST)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(GT, PRED_TEST))
    print('\n')
    
    # Save classification results:

    if save == 'yes':
          
        with open(path_results+classification_idSTRING+'.csv', 'w') as csvfile:
            wtr = csv.writer(csvfile, delimiter=';')
            
            
            wtr.writerow( ["Classification report for classifier %s:\n%s\n"
              % (CLF, metrics.classification_report(GT, PRED_TEST))])
            wtr.writerow( ["Confusion matrix:\n%s" % metrics.confusion_matrix(GT, PRED_TEST)])
            wtr.writerow( [" Accuracy: \n%s" % metrics.accuracy_score(GT, PRED_TEST)])
            wtr.writerow( [" Area under Curve: \n%s" % metrics.roc_auc_score(GT, PRED_TEST_PROBA, average='macro', sample_weight=None)])
            wtr.writerow( ["Predictions: \n"])
            for p in range(len(PRED_TEST)):
                wtr. writerow([testFilenames[p] +'    GT: ' +str(GT[p])+ '   PRED_TEST: '+ str(PRED_TEST[p])])
                
    # append results to summary file:  
    with open(summary_filename, 'a') as summaryFile:
        writer=csv.writer(summaryFile, delimiter=',')
        
        
        #compute parameters to show:
        accuracy=metrics.accuracy_score(GT, PRED_TEST)
        AccuracyTRAIN=metrics.accuracy_score(GT_TRAIN, PRED_TRAIN)
        try: 
            gtACTIVEpACTIVE, gtMQUEENpACTIVE, gtACTIVEpMQUEEN, gtMQUEENpMQUEEN=metrics.confusion_matrix(GT, PRED_TEST).ravel()
        except ValueError as e:
            if sum(PRED_TEST)==len(PRED_TEST) and sum(GT)==len(PRED_TEST) :
                gtACTIVEpACTIVE=0 
                gtMQUEENpACTIVE=0
                gtACTIVEpMQUEEN=0
                gtMQUEENpMQUEEN =len(PRED_TEST)
            
            elif sum(PRED_TEST)==0 and sum(GT)==0 :
                gtACTIVEpACTIVE=len(PRED_TEST)
                gtMQUEENpACTIVE=0
                gtACTIVEpMQUEEN=0
                gtMQUEENpMQUEEN =0
            
            elif sum(PRED_TEST)==len(PRED_TEST) and sum(GT)==0 :
                gtACTIVEpACTIVE=0
                gtMQUEENpACTIVE=0
                gtACTIVEpMQUEEN=len(PRED_TEST)
                gtMQUEENpMQUEEN =0
                
            elif sum(PRED_TEST)==0 and sum(GT)==len(GT) :
                gtACTIVEpACTIVE=0
                gtMQUEENpACTIVE=len(PRED_TEST)
                gtACTIVEpMQUEEN=0
                gtMQUEENpMQUEEN =0
                
                
                
        try: 
            gtACTIVEpACTIVE_TRAIN, gtMQUEENpACTIVE_TRAIN, gtACTIVEpMQUEEN_TRAIN, gtMQUEENpMQUEEN_TRAIN=metrics.confusion_matrix(GT_TRAIN, PRED_TRAIN).ravel()
        except ValueError as e:
            if sum(PRED_TRAIN)==len(PRED_TRAIN) and sum(GT_TRAIN)==len(PRED_TRAIN) :
                gtACTIVEpACTIVE_TRAIN=0 
                gtMQUEENpACTIVE_TRAIN=0
                gtACTIVEpMQUEEN_TRAIN=0
                gtMQUEENpMQUEEN_TRAIN =len(PRED_TEST)
            
            elif sum(PRED_TRAIN)==0 and sum(GT_TRAIN)==0 :
                gtACTIVEpACTIVE_TRAIN=len(PRED_TRAIN)
                gtMQUEENpACTIVE_TRAIN=0
                gtACTIVEpMQUEEN_TRAIN=0
                gtMQUEENpMQUEEN_TRAIN =0
            
            elif sum(PRED_TRAIN)==len(PRED_TRAIN) and sum(GT_TRAIN)==0 :
                gtACTIVEpACTIVE_TRAIN=0
                gtMQUEENpACTIVE_TRAIN=0
                gtACTIVEpMQUEEN_TRAIN=len(PRED_TRAIN)
                gtMQUEENpMQUEEN_TRAIN =0
                
            elif sum(PRED_TRAIN)==0 and sum(GT_TRAIN)==len(GT_TRAIN) :
                gtACTIVEpACTIVE_TRAIN=0
                gtMQUEENpACTIVE_TRAIN=len(PRED_TRAIN)
                gtACTIVEpMQUEEN_TRAIN=0
                gtMQUEENpMQUEEN_TRAIN =0  
                
        try:
            ShannonEnthropy_TRAIN=-(((gtACTIVEpACTIVE_TRAIN+gtACTIVEpMQUEEN_TRAIN)/(gtACTIVEpACTIVE_TRAIN+gtMQUEENpACTIVE_TRAIN+gtACTIVEpMQUEEN_TRAIN+gtMQUEENpMQUEEN_TRAIN) )*log(((gtACTIVEpACTIVE_TRAIN+gtACTIVEpMQUEEN_TRAIN)/(gtACTIVEpACTIVE_TRAIN+gtMQUEENpACTIVE_TRAIN+gtACTIVEpMQUEEN_TRAIN+gtMQUEENpMQUEEN_TRAIN) ))  +    ((gtMQUEENpACTIVE_TRAIN+gtMQUEENpMQUEEN_TRAIN)/(gtACTIVEpACTIVE_TRAIN+gtMQUEENpACTIVE_TRAIN+gtACTIVEpMQUEEN_TRAIN+gtMQUEENpMQUEEN_TRAIN) )*log(((gtMQUEENpACTIVE_TRAIN+gtMQUEENpMQUEEN_TRAIN)/(gtACTIVEpACTIVE_TRAIN+gtMQUEENpACTIVE_TRAIN+gtACTIVEpMQUEEN_TRAIN+gtMQUEENpMQUEEN_TRAIN) )))
        except Exception as e :
            ShannonEnthropy_TRAIN=0
                
        try:
            Precision_on_MQUEEN=gtMQUEENpMQUEEN/(gtMQUEENpMQUEEN+gtACTIVEpMQUEEN)
        except ZeroDivisionError as e:
            Precision_on_MQUEEN=0
        try:
            Recall_on_MQUEEN=gtMQUEENpMQUEEN/(gtMQUEENpMQUEEN+gtMQUEENpACTIVE)
        except ZeroDivisionError as e:
            Recall_on_MQUEEN=0
        try:
            Precision_on_ACTIVE=gtACTIVEpACTIVE/(gtACTIVEpACTIVE+gtMQUEENpACTIVE)
        except ZeroDivisionError as e:
            Precision_on_ACTIVE=0
        try:
            Recall_on_ACTIVE=gtACTIVEpACTIVE/(gtACTIVEpACTIVE+gtACTIVEpMQUEEN)        
        except ZeroDivisionError as e:
            Recall_on_ACTIVE=0
            
        try:
            ShannonEnthropy=-(((gtACTIVEpACTIVE+gtACTIVEpMQUEEN)/(gtACTIVEpACTIVE+gtMQUEENpACTIVE+gtACTIVEpMQUEEN+gtMQUEENpMQUEEN) )*log(((gtACTIVEpACTIVE+gtACTIVEpMQUEEN)/(gtACTIVEpACTIVE+gtMQUEENpACTIVE+gtACTIVEpMQUEEN+gtMQUEENpMQUEEN) ))  +    ((gtMQUEENpACTIVE+gtMQUEENpMQUEEN)/(gtACTIVEpACTIVE+gtMQUEENpACTIVE+gtACTIVEpMQUEEN+gtMQUEENpMQUEEN) )*log(((gtMQUEENpACTIVE+gtMQUEENpMQUEEN)/(gtACTIVEpACTIVE+gtMQUEENpACTIVE+gtACTIVEpMQUEEN+gtMQUEENpMQUEEN) )))
        except Exception as e :
            ShannonEnthropy=0
            
        if ShannonEnthropy>0.9:
            accuracy_on_balancedDatasets=accuracy
        else: 
            accuracy_on_balancedDatasets=0
        
        try:
            AUC_TRAIN=metrics.roc_auc_score(GT_TRAIN, PRED_TRAIN_PROBA, average='macro', sample_weight=None)
        except Exception as e:
            AUC_TRAIN='error'

        try: 
            AUC_TEST=metrics.roc_auc_score(GT, PRED_TEST_PROBA, average='macro', sample_weight=None)
            
        except Exception as e:
            AUC_TEST='error'
            
            
        writer.writerow([classification_idSTRING, AccuracyTRAIN, AUC_TRAIN, gtACTIVEpACTIVE_TRAIN, gtMQUEENpACTIVE_TRAIN, gtACTIVEpMQUEEN_TRAIN, gtMQUEENpMQUEEN_TRAIN , ShannonEnthropy_TRAIN, accuracy , AUC_TEST ,metrics.confusion_matrix(GT, PRED_TEST),Precision_on_MQUEEN , Recall_on_MQUEEN, Precision_on_ACTIVE, Recall_on_ACTIVE, gtACTIVEpACTIVE, gtMQUEENpACTIVE, gtACTIVEpMQUEEN,gtMQUEENpMQUEEN, ShannonEnthropy,accuracy_on_balancedDatasets])
        

        
#TODO:        
        
# make classification at audio-file level (total 10 min ---> concatenating averages of 1 min slices, as BIZOT's paper.)        
#def 
      

# def process raw features to pair mel and HT as different channels of an image   ? 




#def balance_set(sample_set_ids, labels_file)
# CNN functions:




#prepare data to feed CNN: A: feature extraction  1) each sample is a melspectrogram, 2) each sample is a 2 channel image (mel spectrogram + HHT), 3) averages over time each sample 4) stack averages for 10 min together as a single sample.  --> label? will be the majority?



    
    
    
