#
# utility functions
import glob
import os
import librosa
import pdb
import csv
import json
import re
import numpy as np
import random
from info import i, printb, printr, printp, print






# functions to deal with annotations .lab files:

def read_beeNotBee_annotations_saves_labels(audiofilename, block_name,  blockStart, blockfinish, annotations_path, threshold=0):
    
    
    ## function: reads corresponding annotation file (.lab) and assigns a label to one block/sample. Appends label into csv file.
    ##
    ## inputs: 
    ## audiofilename = name of the audio file (no path), block_name = name of the sample/segment,  blockStart = time point in seconds where block starts, blockfinish = time point in seconds where block ends, annotations_path = path to annotations folder (where .lab files are), threshold = value tor threshold. 
    ##
    ## outputs:
    ## label_th= 2 element list, [0] = a label (bee / nobee) for the block and threshold considered; [1] = label strength, value that reflects the proportion of nobee interval in respect to the whole block.
    
    
    # thershold gives the minimum duration of the no bee intervals we want to consider.
    # trheshold=0 uses every event as notBee whatever the duration
    # thershold=0.5 disregards intervals with less than half a second duration.
    
    block_length=blockfinish-blockStart
    
    if audiofilename.startswith('#'):
        annotation_filename=audiofilename[1:-4]+'.lab'
    else :
        annotation_filename=audiofilename[0:-4]+'.lab'
        
        
    try:    
        with open(annotations_path + os.sep + annotation_filename,'r') as f:
            # EXAMPLE FILE:
            
            # CF003 - Active - Day - (223)
            # 0	8.0	bee
            # 8.01	15.05	nobee
            # 15.06	300.0	bee 
            # .
            #
            
            # all files end with a dot followed by an empty line.

            print(annotations_path + os.sep + annotation_filename)
            lines = f.read().split('\n')
        
            labels_th=['bee', 0.0]
            label2assign='bee'
            label_strength=0
            intersected_s=0
                            
            for line in lines:
                if (line == annotation_filename[0:-4]) or (line == '.') or (line ==''):
                    #ignores title, '.', or empty line on the file.
                    continue
                
                #print(line)
                parsed_line= line.split('\t')    
                
                assert (len(parsed_line)==3), ('expected 3 fields in each line, got: '+str(len(parsed_line))) 
                #pdb.set_trace()
                
                tp0=float(parsed_line[0])
                tp1=float(parsed_line[1])
                annotation_label=parsed_line[2]
                if blockfinish < tp0: # no need to read further nobee intervals since annotation line is already after bloch fininshes
                    break
                    
                if annotation_label== 'nobee':
                    
                        
                    if tp1-tp0 >= threshold:  # only progreess if nobee interval is longer than defined threshold.
                    
                        if tp0 > blockStart and tp0 <= blockfinish and tp1 >= blockfinish:
                            
                            intersected_s=intersected_s + (blockfinish-tp0)    
                            # |____________########|########
                            # bs          tp0      bf      tp1 
                        
                        elif tp1 >= blockStart and tp1 < blockfinish and tp0 <= blockStart:
                            
                            intersected_s=intersected_s+ (tp1-blockStart)
                            # #####|########_____|
                            # tp0  bs     tp1    bf
                            
                            
                        elif tp1 >= blockStart and tp1 <= blockfinish and tp0 >= blockStart and tp0 <= blockfinish:
                            
                            intersected_s=intersected_s+ (tp1-tp0)
                            # |_____########_____|
                            # bs   tp0    tp1    bf
                        
                        elif tp0 <= blockStart and tp1 >= blockfinish:
                            
                            intersected_s=intersected_s + (blockfinish-blockStart)
                            #  ####|############|####
                            # tp0  bs           bf  tp1
                            
                    if intersected_s > 0:
                        label2assign='nobee'
                    label_strength= intersected_s/block_length # proportion of nobee length in the block
                    
                    
                    labels_th= [label2assign, round(label_strength,2)]  # if label_strehgth ==0 --> bee segment 
                    
                    assert (blockfinish <=tp1 ), ('the end of the request block falls outside the file: block ending: '+ str(blockfinish)+' end of file at: '+ str(tp1))
            
    except FileNotFoundError as e:
        print(e, '--Anotation file does not exist! label as unknown')
        print(annotation_filename=audiofilename[0:-4]+'.lab')
            
        label2assign = 'unknown'
        label_strength=-1
        
        labels_th = [label2assign, label_strength]
            
    except Exception as e1:
        print('unknown exception: '+str(e1))
        quit
        
    return labels_th


    
#pre_processing functions:



def uniform_block_size(undersized_block, block_size_samples, method='repeat' ):

    lengthTofill=(block_size_samples)-(undersized_block.size)
    if method == 'zero_padding':
        new_block=np.pad(undersized_block, (0,lengthTofill), 'constant', constant_values=(0) )

    elif method=='mean_padding':
        new_block=np.pad(undersized_block, (0,lengthTofill), 'mean' )
    
    elif method=='repeat':        
        new_block= np.pad(undersized_block, (0,lengthTofill), 'reflect')
    else:
        print('methods to choose are: \'zero_padding\' ,\'mean_padding\' and \'repeat\' ' )
        new_block=0
              
    return new_block
    
    
    


     

def load_audioFiles_saves_segments( path_audioFiles,path_save_audio_labels, block_size , thresholds, annotations_path, save_audioSegments='yes'):

    
    audiofilenames_list = [os.path.basename(x) for x in glob.glob(path_audioFiles+'*.mp3')]
    audiofilenames_list.extend([os.path.basename(x) for x in glob.glob(path_audioFiles+'*.wav')])
    
    fi=0
    for file_name in audiofilenames_list:
        fi=fi+1
        print('\n')
        printb('Processing '+ file_name+'          :::file number:  '+str(fi)+' --------->of '+str(len(audiofilenames_list)))
          

        offset=0
        block_id =0
        
        
        while 1:
                    
            # READ ONE BLOCK OF THE AUDIO FILE
            try:
                block,sr = librosa.core.load(path_audioFiles+file_name, offset=offset, duration=block_size)
                print('-----------------Reading segment '+str(block_id))
            except ValueError as e:
                e
                if 'Input signal length' in str(e):
                    block=np.arange(0)
            except FileNotFoundError as e1:
                print(e1, ' but continuing anyway')
                

            if block.shape[0] > 0:    #when total length = multiple of blocksize, results that last block is 0-lenght, this if bypasses those cases.
                
                block_name=file_name[0:-4]+'__segment'+str(block_id)
                print(block_name)
                
                # READ BEE NOT_BEE ANNOTATIONS:
                blockStart=offset
                blockfinish=offset+block_size
                
                for th in thresholds:
                
                    label_file_exists = os.path.isfile(path_save_audio_labels+'labels_BeeNotBee_th'+str(th)+'.csv')
                    with open(path_save_audio_labels+'labels_BeeNotBee_th'+str(th)+'.csv','a', newline='') as label_file:
                        writer =csv.DictWriter(label_file, fieldnames=['sample_name', 'segment_start','segment_finish', 'label', 'label_strength'], delimiter=',')
                        if not label_file_exists:
                            writer.writeheader()
                        
                        label_block_th=read_beeNotBee_annotations_saves_labels(file_name, block_name,  blockStart, blockfinish, annotations_path, th)
                                                   
                        writer.writerow({'sample_name': block_name, 'segment_start': blockStart, 'segment_finish': blockfinish,  'label': label_block_th[0] , 'label_strength': label_block_th[1]})
                        print('-----------------Wrote label for th '+ str(th)+' seconds of segment'+str(block_id)  ) 
                
               
                # MAKE BLOCK OF THE SAME SIZE:
                if block.shape[0] < block_size*sr:   
                    block = uniform_block_size(block, block_size*sr, 'repeat')
                    print('-----------------Uniformizing block length of segment'+str(block_id)  ) 

                        
            
                # Save audio segment:
                if save_audioSegments=='yes' and (not os.path.exists(path_save_audio_labels+block_name+'.wav')): #saves only if option is chosen and if block file doesn't already exist.
                    librosa.output.write_wav(path_save_audio_labels+block_name+'.wav', block, sr)
                    print( '-----------------Saved wav file for segment '+str(block_id))
                
                    
                    
            else :
                print('----------------- no more segments for this file--------------------------------------')
                print('\n')
                break
            offset += block_size
            block_id += 1
    printb('______________________________No more audioFiles___________________________________________________')
       
    return 

    
def read_HiveState_fromSampleName( filename, states):   #states: state_labels=['active','missing queen','swarm' ]
    label_state='other'
    for state in states:
        if state in filename.lower():
            label_state = state
    #incorporate condition for italian recordings which do not follow the same annotation: 'QueenBee' or 'NO_QueenBee'        
    if label_state=='other':
        if 'NO_QueenBee' in filename:
            label_state = states[1]
        else:
            label_state=states[0]
    return label_state
    
def write_Statelabels_from_beeNotBeelabels(path_save, path_labels_BeeNotBee, states=['active','missing queen','swarm' ]):
    
    label_file_exists = os.path.isfile(path_save+'state_labels.csv')
    
    with open(path_labels_BeeNotBee, 'r' ) as rfile, \
    open(path_save+'state_labels.csv', 'a', newline='') as f_out:
        csvreader = csv.reader(rfile, delimiter=',')
        writer= csv.DictWriter(f_out, fieldnames=['sample_name', 'label'], delimiter=',') 
        
        if not label_file_exists:
            writer.writeheader()
        
        for row in csvreader:
            if not row[0]=='sample_name':
                label_state=read_HiveState_fromSampleName(row[0], states)
                writer.writerow({'sample_name':row[0], 'label':label_state})
    return

def get_list_samples_names(path_audio_folder):
    sample_ids=[os.path.basename(x) for x in glob.glob(path_audio_folder+'*.wav')]
    return sample_ids
    

def write_sample_ids_perHive(sample_ids , savepath):
    
  #identify different hives:
            #in the NU-Hive dataset the hives are identified in the string Hive1 or Hive3 in the beginning.
            #in OSBH every file referring to the same person will be considered as if the same Hive: identified by #nameInitials -
            # other files that do not follow this can be grouped in the same hive ()
            #get from unique filenames all unique identifiers of hives: either read the string until the first_  : example 'Hive3_
            #or get the string starting in '#' until the first ' - '. example: '#CF003 - '
    #uniqueFilenames=['Hive3_20_07_2017_QueenBee_H3_audio___15_40_00.wav', 
    #                 'Hive1_20_07_2017_QueenBee_H3_audio___15_40_00.wav', 
    #                 'Hive3_20_07_22017_QueenBee_H3_audio___15_40_00.wav', 
    #                 'Sound Inside a Swarming Bee Hive  -25 to -15 minutes-sE02T8B2LfA.wav', 
    #                 'Sound Inside a Swarming Bee Hive  +25 to -15 minutes-sE02T8B2LfA.wav', 
    #                 '#CF003 - Active - Day - (222).csv', '#CF003 - Active - Day - (212).csv']
    
    
    uniqueHivesNames={}
    pat1=re.compile("(\w+\d)\s-\s")
    pat2=re.compile("^(Hive\d)_")
    for sample in sample_ids:

        match_pat1=pat1.match(sample)
        match_pat2=pat2.match(sample)
        if match_pat1:
            if match_pat1.group(1) in uniqueHivesNames.keys():
                uniqueHivesNames[match_pat1.group(1)].append(sample)
            else: 
                uniqueHivesNames[match_pat1.group(1)]=[sample]
        elif match_pat2:
            if match_pat2.group(1) in uniqueHivesNames.keys():
                uniqueHivesNames[match_pat2.group(1)].append(sample)
            else: 
                uniqueHivesNames[match_pat2.group(1)]=[sample]
        else: 
            #odd case, like files names 'Sound Inside a Swarming Bee Hive  -25 to -15 minutes-sE02T8B2LfA.wav'
             #will be all gatehered as the same hive, although we need to be careful if other names appear!
            if 'Sound Inside a Swarming Bee Hive' in uniqueHivesNames.keys():
                uniqueHivesNames['Sound Inside a Swarming Bee Hive'].append(sample)
            else: 
                uniqueHivesNames['Sound Inside a Swarming Bee Hive']=[sample]  
                
    
    
    with open(savepath+'sampleID_perHive.json', 'w') as outfile:
        json.dump(uniqueHivesNames, outfile)
    
    return uniqueHivesNames
    
    
def get_uniqueHives_names_from_File(path_file_samplesId_perHive):
    
    hives_data = json.load(open(path_file_samplesId_perHive + 'sampleID_perHive.json', "r"))

    n_hives=len(hives_data.keys())
    
    return hives_data
     

     

   
#todo

#def balance_dataset()
# splitBy = 'day'
# splitBy = 'hive'
# splitBy = 'dataset'
# splitBy = 'location'
# splitBy = 'randomly'
    
def split_samples_byHive(test_size, train_size, hives_data_dictionary, splitPath_save):
    
    ## creates 3 different sets intended for hive-independent classification. meaning that samples are separated accordingly to the hive.
    ## input: test_size, ex: 0.1  : 10% hives for test
    ## train_size, ex: 0.7: 70% hives for training, 30% for valisdation. (after having selected test samples!!)  
    ## splitPath_save = path and name where to save the splitted samples id dictionary
    
    ## output:
    ## returns and dumps a dictionary: {test : [sample_id1, sample_id2, ..], train : [], 'val': [sample_id2, sample_id2]}
    
    splittedSamples={'test': [], 'train': [], 'val':[]}
    
    n_hives = len(hives_data_dictionary.keys())
            
    hives_list=list(hives_data_dictionary.keys())
    
    
    
    hives_rest1=random.sample(hives_list, round(n_hives*(1-test_size)))
    
    if len(hives_rest1) == len(hives_list):
        rand_hive = random.sample(range(len(hives_rest1)),1)
        hives_rest=hives_rest1[:]
        del hives_rest[rand_hive[0]]
    else:
        
        hives_rest = hives_rest1[:]
  
    hiveTEST=np.setdiff1d(hives_list , hives_rest)
    hiveVAL=random.sample(hives_rest, round(len(hives_rest)*train_size))
    hiveTRAIN=np.setdiff1d(hives_rest , hiveVAL)
    
    
    print('hives for testing: '+ str(list(hiveTEST)))
    print('hives for training: '+ str(list(hiveTRAIN)))
    print('hives for validation: '+ str(hiveVAL))
    
    
    for ht in list(hiveTEST):
        splittedSamples['test'].extend(hives_data_dictionary[ht])

    for h1 in list(hiveTRAIN):
        splittedSamples['train'].extend(hives_data_dictionary[h1])
    
    for h2 in hiveVAL:
        splittedSamples['val'].extend(hives_data_dictionary[h2])

    with open(splitPath_save+'.json', 'w') as outfile:
        json.dump(splittedSamples, outfile)
    
    return splittedSamples



        
    
    
    #return index_TESTE, index_TRAIN, index_VAL


    #save ids for csv or similar! maybe directly for numpy archive, or dump
    
    # return test_ids, train_ids, val_ids
    
# feature extraction 
#  SVM functions

    
    