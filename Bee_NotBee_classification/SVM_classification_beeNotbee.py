import os

import numpy as np
from info import i, printb, printr, printp, print
from utils import load_audioFiles_saves_segments, write_Statelabels_from_beeNotBeelabels, get_uniqueHives_names_from_File, split_samples_byHive, get_samples_id_perSet, get_features_from_samples, get_GT_labels_fromFiles, labels2binary



def main():
   #----------------------------------- parameters to change-----------------------------------#
    path_workingFolder='E:\\dataset_BeeNoBee_'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
    labels2read= 'labels_BeeNotBee_th0'
    feature = 'MFCCs20'
   #-------------------------------------------------------------------------------------------#
    


#Feature extraction: 
    sample_ids_test, sample_ids_train, sample_ids_val = get_samples_id_perSet(path_workingFolder+'split_random_0.json') 
    
    X_train = get_features_from_samples(path_workingFolder, sample_ids_train, 'MFCCs20', 'NO', 1)
    X_val = get_features_from_samples(path_workingFolder, sample_ids_val, 'MFCCs20', 'NO', 1)
    X_test = get_features_from_samples(path_workingFolder, sample_ids_test, 'MFCCs20', 'NO', 1)
    
    
    labels_train = get_GT_labels_fromFiles(path_workingFolder, sample_ids_train, labels2read)
    Y_train= labels2binary('nobee', labels_train)
    
    labels_val = get_GT_labels_fromFiles(path_workingFolder, sample_ids_val, labels2read)
    Y_val= labels2binary('nobee', labels_val)
    
    labels_test = get_GT_labels_fromFiles(path_workingFolder, sample_ids_test, labels2read)
    Y_test= labels2binary('nobee', labels_test)
    
    
    X_flat_train=np.concatenate( X_train , X_val)
    
    
    clf, y_pred_proba_train, y_pred_train, y_pred_proba_test, y_pred_test= SVM_Classification_inSplittedSets(X_flat_train, Y_train, X_test, Y_test, kerneloption='rbf')
    
    
    
if __name__ == "__main__":
    main()