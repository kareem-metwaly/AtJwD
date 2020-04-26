# POST-PROCESSING IRCNN 
The folder include all the post-processing codes, saved models and tesing results prepared for NTIRE 2020 Dehazing Challenge submitted from iPAL-NonLocal
(url: https://competitions.codalab.org competitions/22236)

Post processing DH-IRCNN for dehazed images
Team name: iPAL-NonLocal
Team leader: Kareem Metwaly (metwaly.kareem@gmail.com)
Team members:  Kareem Metwaly, Xuelu Li, Tiantong Guo, Vishal Monga 
Time: 03/26/2020

## NOTE 
This is not the proposed dehazing model. This code serves as the post-processing for the proposed model's output. The IRCNN uses the denoiser trained on Sigam=15 to reduce the artifacts of the dehazed image and improve the image quality. 

This code and the model in the DH-IRCNN *only* produce the denoised version of the outputs from the proposed model. 


## PREREQUISITE 
To correctly run the test model, the following packages are required.
1. MATLAB, version>2016a
2. matconvnet for MATLAB (included in the folder; for installation please follow: http://www.vlfeat.org/matconvnet/install/)
3. the results from different proposed method stored in the 'testsets/' folder

## TO RUN 
Run the following file in MATLAB after install the matconvnet:
	MAINTHREAD_DH_IRCNN.m

## FILES 
1. MAINTHREAD_DH_IRCNN.m:
The file includes post-processing codes for de-noising the outputs from our proposed network.

2. testsets:
The folder includes the outputs from our proposed model stored in separate folders:
	(1) AtJw: include the test results from "AtJw" Dehazing Network.
	(3) AtJw+D: include the test results from "AtJw+D" Dehazing Network.

3. results: 
The folder includes the post-processed results using DH-IRCNN denoiser stored in separate folders:
	(1) DH-IRCNN_AtJw: include the denoised results of AtJw_DH
	(3) DH-IRCNN_AtJw+D: include the denoised results of AtJw+D_DH

4. models: 
The folder includes the color model used by DH-IRCNN denoiser.

5. kernels:
The folder includes the kernels used by DH-IRCNN denoiser

6. utilities:
The folder includes small snippets that can be used though the application.

7. matconvnet-1.0-beta25:
The pre-included matconvnet for MATLAB for easy usage and installation.
