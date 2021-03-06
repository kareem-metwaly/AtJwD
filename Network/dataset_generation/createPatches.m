clc;
close all;
clear variables;

inputDirectory = './GT/';
hazeDirectory = './HAZY/';

trainingDirectory_data = './training_patches_varied_256/input/';
trainingDirectory_haze = './training_patches_varied_256/haze/';

patchSize = 256;
sizeBy2 = patchSize/2;
stride = 128;

files_input = dir([inputDirectory]);
files_hazy = dir([hazeDirectory]);

files_input=files_input(~ismember({files_input.name},{'.','..'}));
files_hazy=files_hazy(~ismember({files_hazy.name},{'.','..'}));

% files_input = dir([inputDirectory, '*.png']);
% files_hazy = dir([hazeDirectory, '*.png']);

length_files = length(files_input);
delta = .01;
% input_image = imread([inputDirectory, files_input(1).name]);


string_input = 'input';
string_haze = 'haze';

k = 1;
for i = 1: length_files
    disp(i)
    input_image = imread([inputDirectory, files_input(i).name]);
    haze_image = imread([hazeDirectory, files_hazy(i).name]);
    input_image  = double(input_image);
    haze_image  = double(haze_image);
    input_image = input_image/255;
    haze_image = haze_image/255;
    
    [sizeX, sizeY, sizeZ] = size(input_image);


    startX = patchSize;
    endX = sizeX - patchSize;
    startY = patchSize;
    endY = sizeY - patchSize;
    
    cropped_input = imresize(input_image,[patchSize patchSize]);
    cropped_haze = imresize(haze_image,[patchSize patchSize]);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameHaze = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input, filenameData);
    imwrite(cropped_haze, filenameHaze);
    k = k+1;
    
    cropped_input_flip = fliplr(cropped_input);
    cropped_haze_filp = fliplr(cropped_haze);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input_flip, filenameData);
    imwrite(cropped_haze_filp, filenameLabel);
    k = k+1;
    
    cropped_input_rot = imrotate(cropped_input,90);
    cropped_haze_rot = imrotate(cropped_haze, 90);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input_rot, filenameData);
    imwrite(cropped_haze_rot, filenameLabel);
    k = k+1;
    
    cropped_input_rot = imrotate(cropped_input,180);
    cropped_haze_rot = imrotate(cropped_haze, 180);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input_rot, filenameData);
    imwrite(cropped_haze_rot, filenameLabel);
    k = k+1;
    
    cropped_input_rot = imrotate(cropped_input,270);
    cropped_haze_rot = imrotate(cropped_haze, 270);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input_rot, filenameData);
    imwrite(cropped_haze_rot, filenameLabel);
    k = k+1;
    
    cropped_input_rot = imrotate(cropped_input_flip,90);
    cropped_haze_rot = imrotate(cropped_haze_filp, 90);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input_rot, filenameData);
    imwrite(cropped_haze_rot, filenameLabel);
    k = k+1;
    
    cropped_input_rot = imrotate(cropped_input_flip,180);
    cropped_haze_rot = imrotate(cropped_haze_filp, 180);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input_rot, filenameData);
    imwrite(cropped_haze_rot, filenameLabel);
    k = k+1;
    
    cropped_input_rot = imrotate(cropped_input_flip,270);
    cropped_haze_rot = imrotate(cropped_haze_filp, 270);
    filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
    filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
    imwrite(cropped_input_rot, filenameData);
    imwrite(cropped_haze_rot, filenameLabel);
    k = k+1;
    
    
    input_image = padarray(input_image, [sizeBy2,sizeBy2]);
    haze_image = padarray(haze_image, [sizeBy2,sizeBy2]);
    
    for x = startX:stride:endX
        for y = startY:stride:endY
            input_patch = input_image(x-sizeBy2+1:x+sizeBy2, y-sizeBy2+1:y+sizeBy2,:);
            haze_patch = haze_image(x-sizeBy2+1:x+sizeBy2, y-sizeBy2 + 1:y+sizeBy2, :);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameHaze = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(input_patch, filenameData);
            imwrite(haze_patch, filenameHaze);
            k = k+1;
            
            cropped_input_flip = fliplr(input_patch);
            cropped_haze_filp = fliplr(haze_patch);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(cropped_input_flip, filenameData);
            imwrite(cropped_haze_filp, filenameLabel);
            k = k+1;
            
            cropped_input_rot = imrotate(input_patch,90);
            cropped_haze_rot = imrotate(haze_patch, 90);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(cropped_input_rot, filenameData);
            imwrite(cropped_haze_rot, filenameLabel);
            k = k+1;
            
            cropped_input_rot = imrotate(input_patch,180);
            cropped_haze_rot = imrotate(haze_patch, 180);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(cropped_input_rot, filenameData);
            imwrite(cropped_haze_rot, filenameLabel);
            k = k+1;
            
            cropped_input_rot = imrotate(input_patch,270);
            cropped_haze_rot = imrotate(haze_patch, 270);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(cropped_input_rot, filenameData);
            imwrite(cropped_haze_rot, filenameLabel);
            k = k+1;
            
            
            cropped_input_rot = imrotate(cropped_input_flip,90);
            cropped_haze_rot = imrotate(cropped_haze_filp, 90);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(cropped_input_rot, filenameData);
            imwrite(cropped_haze_rot, filenameLabel);
            k = k+1;
            
            cropped_input_rot = imrotate(cropped_input_flip,180);
            cropped_haze_rot = imrotate(cropped_haze_filp, 180);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(cropped_input_rot, filenameData);
            imwrite(cropped_haze_rot, filenameLabel);
            k = k+1;
            
            cropped_input_rot = imrotate(cropped_input_flip,270);
            cropped_haze_rot = imrotate(cropped_haze_filp, 270);
            filenameData = [trainingDirectory_data,string_input,'_',num2str(k),'.png'];
            filenameLabel = [trainingDirectory_haze,string_haze,'_',num2str(k),'.png'];
            imwrite(cropped_input_rot, filenameData);
            imwrite(cropped_haze_rot, filenameLabel);
            k = k+1;
        end
    end
    
end
