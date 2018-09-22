#
#loads audio files which have annotations (maybe provide only data from the BeeNotBee annotated dataset
#segments and saves chunks accordingly to segment size, keepname of file+_segment_id
#reads both labes from annotations and state, save both csv

#only calls functions, previously declared in utils.py



import os
from info import i, printb, printr, printp, print
from utils import load_audioFiles_saves_segments, write_Statelabels_from_beeNotBeelabels



def main():
   #----------------------------------- parameters to change-----------------------------------#
    block_size=60 # blocks of 60 seconds
    thresholds=[0, 5]  # minimum length for nobee intervals: 0 or 5 seconds (creates one label file per threshold value
    path_audioFiles="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\To Bee or not to Bee_the annotated dataset"+os.sep
    annotations_path="C:\\Users\\ines\\Dropbox\\QMUL\\BEESzzzz\\To Bee or not to Bee_the annotated dataset"+os.sep
    path_save_audio_labels='E:\\dataset_BeeNoBee_'+str(block_size)+'sec'+os.sep
    #-------------------------------------------------------------------------------------------#
    
    
    if not os.path.exists(path_save_audio_labels):
        os.makedirs(path_save_audio_labels)
    
    # segments audio files, assigns label BeeNotBee to each block, writes labels to csv , saves segmened blocks in wav.
    # independently of flag save_audioSegments, if .wav with same name already exists it won't save again.
    # new labels are just appended to existing labels file, if purpose is to redo the whole file delete before running.
    load_audioFiles_saves_segments( path_audioFiles, path_save_audio_labels, block_size , thresholds, annotations_path, save_audioSegments='yes')
    
    
    path_beeNotbee_labels=path_save_audio_labels + 'labels_BeeNotBee_th'+str(thresholds[0])+'.csv'
    # reads labels beeNotBee files and creates corresponding states label file.
    write_Statelabels_from_beeNotBeelabels(path_save_audio_labels, path_beeNotbee_labels, states=['active','missing queen','swarm' ])


if __name__ == "__main__":
    main()