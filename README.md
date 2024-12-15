# ELEC576_Team4_Final_Project
This repository contains a prepared version of our training, testing, and analysis code for our ELEC576 project for Fall 2024.

### Installation
```
git clone https://github.com/masonweiss/ELEC576_Team4_Final_Project.git
```
Training for the models was performed using Google Colab A100 GPUs; the paths have been updated from the team's shared drive format to relative paths from the upload location in any user's Google Drive. In order to run the code included in this repository, we recommend utilizing Google Colab (particularly with GPU access) to run the .ipynb files. 

1. Follow the instructions below to download the dataset used for training and testing. The data should be uploaded to match the file path 'ELEC576_Team4_Final_Project/dataset/paired/...'.
2. Upload this entire repository to your home folder ("MyDrive") in Google Drive.
3. Open any of the .ipynb files (analysis, IR_Former, OurModel) using Google Colab.
4. Mount your google drive using the first cell in whichever notebook you're working with, change directory to the upload location in drive, and (for IR_Former) install the requirements using pip.
5. All cells should work now- as a note, the training functionality expects GPU access so ensure that the hosted runtime provides such.

### Dataset
The paired image dataset used can be found at the following [drive link](https://drive.google.com/drive/folders/1YkoISC_PzTKZOuxMqLFBH506ZbSyTwzO?usp=drive_link). The entire folder named `paired` should be placed inside the `dataset` folder within this repository. 

### Team Members
Leo Bashaw - lb73@rice.edu [Captain] \
Melissa Cantú - msc15@rice.edu \
Laura Jabr - lmj7@rice.edu \
Mason Weiss - mw103@rice.edu


### Reference
The `IRFormer/` directory including our "control"/SOTA model for RGB-to-IR image translation was largely based on the code available at the following [repository](https://github.com/CXH-Research/IRFormer). The README and License from CXH Research are retained in that folder.

