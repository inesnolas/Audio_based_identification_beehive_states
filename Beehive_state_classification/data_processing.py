#script to process raw audio files


import os
import pdb
import sys
import glob

sys.path.append('C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\Audio_based_indentification_beehive_State\\Bee_NotBee_classification')
from info import i, printb, printr, printp, print

from utils import load_audioFiles_saves_segments, write_Statelabels_from_samplesFolder, get_uniqueHives_names_from_File, split_samples_byHive, get_samples_id_perSet, get_features_from_samples, get_GT_labels_fromFiles, labels2binary, get_list_samples_names, write_sample_ids_perHive, split_samples_ramdom

def main():
   #----------------------------------- parameters to change-----------------------------------#
    block_size=60 # blocks of 60 seconds
    path_audioFiles="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\Data\\Ancona_AudioData_BeehiveState"+os.sep  # path to audio files
    path_save_audio_labels='E:\\dataset_BeehiveState_'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
    #-------------------------------------------------------------------------------------------#
    
    
    if not os.path.exists(path_save_audio_labels):
        os.makedirs(path_save_audio_labels)
    
    # segments audio files, saves segmened blocks in wav.
    # independently of flag save_audioSegments, if .wav with same name already exists it won't save again.
    
    load_audioFiles_saves_segments( path_audioFiles, 
    path_save_audio_labels, block_size , [0], '-', read_beeNotBee_annotations='no', save_audioSegments='yes')
    
    
    
    path_samplesFolder = path_save_audio_labels
    path_save = path_samplesFolder 
    # reads names of samples (.wav files) and creates corresponding states label file.
    write_Statelabels_from_samplesFolder(path_save, path_samplesFolder, states=['active','missing queen','swarm' ])

    sample_ids=get_list_samples_names(path_save_audio_labels) # get sample ids from audio segments folder.
    
    # split data by Hive 
    hives=write_sample_ids_perHive(sample_ids , path_save_audio_labels)  # retrieves unique hives names and also writes these to a file
    #hives=get_uniqueHives_names_from_File(path_save_audio_labels)
    for i in range(3):
        split_dict = split_samples_byHive(0.1, 0.5, hives, path_save_audio_labels+'split_byHive_'+str(i))
        
        
    #split data by Files    
    audiofilenames_list = [os.path.basename(x) for x in glob.glob(path_audioFiles+'*.mp3')]
    audiofilenames_list.extend([os.path.basename(x) for x in glob.glob(path_audioFiles+'*.wav')])
    sample_perFiles = write_sample_ids_perFILE(sample_ids, audiofilenames_list, path_save_audio_labels)
    for i in range(3):
        split_dict = split_samples_byFILE(0.1, 0.5, path_save_audio_labels+'split_byFiles_'+str(i))
    
    #split data randomly
    for i in range(3):
        split_dict = split_samples_ramdom(0.1,0.5,path_save_audio_labels, path_save_audio_labels+'split_random_'+str(i))
        
    
if __name__ == "__main__":
    main()