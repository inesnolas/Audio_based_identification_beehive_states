#
#loads audio files which have annotations 
#segments and saves chunks accordingly to segment size, 
#reads labels from annotations and state, save both in csv file

#all functions are defined in utils.py



import os
from info import i, printb, printr, printp, print
from utils import load_audioFiles_saves_segments, write_Statelabels_from_beeNotBeelabels, get_uniqueHives_names_from_File, split_samples_byHive, get_samples_id_perSet, get_features_from_samples, get_GT_labels_fromFiles, labels2binary



def main():
   #----------------------------------- parameters to change-----------------------------------#
    block_size=60 # blocks of 60 seconds
    thresholds=[0, 5]  # minimum length for nobee intervals: 0 or 5 seconds (creates one label file per threshold value)
    path_audioFiles="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\To Bee or not to Bee_the annotated dataset"+os.sep  # path to audio files
    annotations_path="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\To Bee or not to Bee_the annotated dataset"+os.sep # path to .lab files
    path_save_audio_labels='E:\\dataset_BeeNoBee_'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
    #-------------------------------------------------------------------------------------------#
    
    
    if not os.path.exists(path_save_audio_labels):
        os.makedirs(path_save_audio_labels)
    
    # segments audio files, assigns label BeeNotBee to each block, writes labels to csv , saves segmened blocks in wav.
    # independently of flag save_audioSegments, if .wav with same name already exists it won't save again.
    # new labels are just appended to existing labels file, if purpose is to redo the whole file delete before running.
    load_audioFiles_saves_segments( path_audioFiles, path_save_audio_labels, block_size , thresholds, annotations_path, read_beeNotBee_annotations='yes', save_audioSegments='yes')
    
    
    path_beeNotbee_labels=path_save_audio_labels + 'labels_BeeNotBee_th'+str(thresholds[0])+'.csv' 
    # reads labels beeNotBee files and creates corresponding states label file.
    write_Statelabels_from_beeNotBeelabels(path_save_audio_labels, path_beeNotbee_labels, states=['active','missing queen','swarm' ])
    
    
    sample_ids=get_list_samples_names(path_save_audio_labels) # get sample ids from audio segments folder.
    
    # split data by Hive 
    hives=write_sample_ids_perHive(sample_ids , path_save_audio_labels)  # retrieves unique hives names and also writes these to a file
    #hives=get_uniqueHives_names_from_File(path_save_audio_labels)
    for i in range(3):
        split_dict = split_samples_byHive(0.1, 0.5, hives, path_save_audio_labels+'split_byHive_'+str(i))
    
    #split data randomly
    for i in range(3):
        split_dict = split_samples_ramdom(0.1,0.5,path_save_audio_labels, path_save_audio_labels+'split_random_'+str(i))
        
       
    
    
if __name__ == "__main__":
    main()