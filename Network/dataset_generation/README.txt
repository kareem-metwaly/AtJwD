######################## DATA-GENERATION #######################
The folder includes all the data-generation codes that were used to generate the h5 file used for training the network
(url: https://competitions.codalab.org competitions/22236)

Data-generation for dehazing:
Team name: iPAL-NonLocal
Team leader: Kareem Metwaly (kareem@psu.edu)
Team members:  Kareem Metwaly, Xuelu Li, Tiantong Guo, Vishal Monga 
Time: 03/26/2020

############################# NOTE ################################
This is not the proposed dehazing model. 
This code serves as the data-generation for the proposed model's output. 

############################## PREREQUISITE ##################################
To correctly run the data-generation, the following packages are required.
1. MATLAB, version>2016a
2. python3
3. dataset ground truth sould be in 'GT' and the hazy input in 'HAZY'.

################################# TO RUN ####################################
Run the following files in order:
	1. from MATLAB run 'createPatches.m' to generate patches.
	2. from python run 'createH5.py' to generate H5 file out of the previously generated patches. Change the value of total according to the total number of generated patches by the MATLAB code.

################################# FILES ######################################
1. createPatches.m:
The file generates patches from the gives dataset. We default to patches of size 256x256 and we implement data augmentation.

2. createH5.py:
The file generates the H5 file used in training from the patches generated by the MATLAB code. Change the value of total according to the total number of generated patches by the MATLAB code.

3. GT and HAZY:
The folders where the dataset should be given.
