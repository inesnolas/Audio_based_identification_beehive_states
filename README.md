# Audio-based identification of beehive states

The sound produced by bees in a beehive can be a source of many insights regarding the overall state, health conditions, and can even indicate natural phenomena related to the natural life cycle of the hive.

This project's goal is to create a system that can automatically identify different states of a hive based on audio recordings made from inside beehives. 

The proposed pipeline consists in a sequence of two classifiers: The first works as a preselector of relevant samples, like a cleaning system to remove any non bee sound.
The selected samples are then fed to the second classifiers wich makes the decision regarding the state of the hive.



## How to use

This repository is organized into two main folders refering to each part of the project.



Bee not bee classification.[1]
The data used to train this classifier is available at: https://zenodo.org/record/1321278#.W2XswdJKjIU

This is an ongoing project for which this document is being updated accordingly.

## License

This project is licensed under the GPL License - see the [LICENSE.md](LICENSE.md) file for details

