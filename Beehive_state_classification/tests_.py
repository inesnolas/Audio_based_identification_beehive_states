
import os
import pdb
import sys

sys.path.append('C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\Audio_based_indentification_beehive_State\\Bee_NotBee_classification')
from info import i, printb, printr, printp, print

from utils import load_audioFiles_saves_segments, write_Statelabels_from_samplesFolder, get_uniqueHives_names_from_File, split_samples_byHive, get_samples_id_perSet, get_features_from_samples, get_GT_labels_fromFiles, labels2binary, get_list_samples_names, write_sample_ids_perHive, split_samples_ramdom

#----------------------------------- parameters to change-----------------------------------#
block_size=60 # blocks of 60 seconds
#path_audioFiles="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\Data\\Ancona_AudioData_BeehiveState"+os.sep  # path to audio files
path_audioFiles = "C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\Data\\data2test"+os.sep 
path_save_audio_labels='E:\\dataset_BeehiveState_'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
#-------------------------------------------------------------------------------------------#
    

path_samplesFolder = path_save_audio_labels
path_save = path_samplesFolder 
# reads names of samples (.wav files) and creates corresponding states label file.
write_Statelabels_from_samplesFolder(path_save, path_samplesFolder, states=['active','missing queen','swarm' ])

#Feature extraction: 
sample_ids_test, sample_ids_train, sample_ids_val = get_samples_id_perSet(path_save_audio_labels+'split_random_0.json')
X_train = get_features_from_samples(path_save_audio_labels, sample_ids_train, 'MFCCs20', 'NO', 0)

labels2read = 'state_labels'
labels_train = get_GT_labels_fromFiles(path_save_audio_labels, sample_ids_train, labels2read)
Y_train= labels2binary('missing queen', labels_train)

pdb.set_trace()
