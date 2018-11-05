# Audio-based identification of beehive states

The sound produced by bees in a beehive can be a source of many insights regarding the overall state, health conditions, and can even indicate natural phenomena related to the natural life cycle of the hive.

This project's goal is to create a system that can automatically identify different states of a hive based on audio recordings made from inside beehives. 

The proposed pipeline consists in a sequence of two classifiers: The first works as a preselector of relevant samples, like a cleaning system to remove any non bee sound.
The selected samples are then fed to the second classifiers wich makes the decision regarding the state of the hive.



## Contents

This repository is organized into two main folders refering to each mentioned part of the project.

Bee not bee classification contains all code developed in the context of the work in [1], it is designed to to use and process data from the annotated dataset available to download at: https://zenodo.org/record/1321278#.W2XswdJKjIU





This is an ongoing project for which this document is being updated accordingly.

## Usage Example

in Bee not bee classification:

Data preprocessing: 
- Start by downloading all audio files (.mp3, .wav) and annotation files (.lab) from https://zenodo.org/record/1321278#.W2XswdJKjIU .we sugest keeping them in a single folder.
- edit Bee_not_bee_classification/data_processing_beeNotbee.py to modify the relevant paths and parameters. 
- run Bee_not_bee_classification/data_processing_beeNotbee.py which will process the raw data and create: the audio samples, label files that contain for each sample the label bee/nobee and the beehive state.
it also splits the samples in 3 sets (train, test and validation) accordingly to the split scheme and saves files containing the id of the samples in each set.

Classification:


  


## License

This project is licensed under the GPL License - see the [LICENSE.md](LICENSE.md) file for details


[1] I. Nolasco and E. Benetos, “To bee or not to bee: Investigating machine learning approaches to beehive sound recognition”, in Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2018, Accepted.
[2] I. Nolasco, “Audio-based beehive state recognition”,  M.S.thesis, Queen Mary University of London, 2018.
