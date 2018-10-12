#
#loads audio files which have annotations (maybe provide only data from the BeeNotBee annotated dataset
#segments and saves chunks accordingly to segment size, keepname of file+_segment_id
#reads both labes from annotations and state, save both csv

#only calls functions, previously declared in utils.py



import os
import pdb

sys.path.append('C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\Audio_based_indentification_beehive_State\\Bee_NotBee_classification')
from info import i, printb, printr, printp, print

from utils import load_audioFiles_saves_segments, write_Statelabels_from_samplesFolder, get_uniqueHives_names_from_File, split_samples_byHive, get_samples_id_perSet, get_features_from_samples, get_GT_labels_fromFiles, labels2binary



def main():
   #----------------------------------- parameters to change-----------------------------------#
    block_size=60 # blocks of 60 seconds
    thresholds=[0, 5]  # minimum length for nobee intervals: 0 or 5 seconds (creates one label file per threshold value)
    #path_audioFiles="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\To Bee or not to Bee_the annotated dataset"+os.sep  # path to audio files
    path_audioFiles="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\To Bee or not to Bee_the annotated dataset"+os.sep  # path to audio files
    path_save_audio_labels='E:\\dataset_BeehiveState_'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
    #-------------------------------------------------------------------------------------------#
    
    
    if not os.path.exists(path_save_audio_labels):
        os.makedirs(path_save_audio_labels)
    
    # segments audio files, saves segmened blocks in wav.
    # independently of flag save_audioSegments, if .wav with same name already exists it won't save again.
    
    load_audioFiles_saves_segments( path_audioFiles, path_save_audio_labels, block_size , thresholds, '-', read_beeNotBee_annotations='no', save_audioSegments='yes')
    
    
    
    path_samplesFolder = path_save_audio_labels
    path_save = path_samplesFolder 
    # reads names of samples (.wav files) and creates corresponding states label file.
    write_Statelabels_from_samplesFolder(path_save, path_samplesFolder, states=['active','missing queen','swarm' ])
    
    pdb.set_trace()
    sample_ids=get_list_samples_names(path_save_audio_labels) # get sample ids from audio segments folder.
    
    # split data by Hive 
    hives=write_sample_ids_perHive(sample_ids , path_save_audio_labels)  # retrieves unique hives names and also writes these to a file
    #hives=get_uniqueHives_names_from_File(path_save_audio_labels)
    for i in range(3):
        split_dict = split_samples_byHive(0.1, 0.5, hives, path_save_audio_labels+'split_byHive_'+str(i))
    
    #split data randomly
    for i in range(3):
        split_dict = split_samples_ramdom(0.1,0.5,path_save_audio_labels, path_save_audio_labels+'split_random_'+str(i))
        
    pdb.set_trace()    
    #Feature extraction: 
    sample_ids_test, sample_ids_train, sample_ids_val = get_samples_id_perSet(path_save_audio_labels+'split_random_0.json')
    X_train = get_features_from_samples(path_save_audio_labels, sample_ids_train, 'MFCCs20', 'NO', 0)
    
    labels2read = 'state_labels'
    labels_train = get_GT_labels_fromFiles(path_save_audio_labels, sample_ids_train, labels2read)
    Y_train= labels2binary('missing queen', labels_train)
    
    
    pdb.set_trace()
    


if __name__ == "__main__":
    main()