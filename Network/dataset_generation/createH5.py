from __future__ import print_function
import cv2
import numpy as np
import glob, os
#import matplotlib.pyplot as plt
import sys
#import time
import h5py
import random
#from scipy import ndimage
import ntpath



DATA_PATH = './training_patches_varied_256/input/'
LABEL_PATH = './training_patches_varied_256/haze/'
PATCH_PATH = './'
SIZE_INPUT = 256
SIZE_TARGET = 256
STRIDE = 128
count = 0
i = 1
total = 39240

h5fw = h5py.File(str(PATCH_PATH + str(SIZE_INPUT) + str(total) + '_' + 'training' + '.h5'), 'w')

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

INPUT = np.empty(shape=(total, SIZE_INPUT, SIZE_INPUT, 3))
TARGET = np.empty(shape=(total, SIZE_TARGET, SIZE_TARGET, 3))

k = 0
p = np.random.permutation(total)
print(p)
for data_image in glob.glob(DATA_PATH + '*.png'):
    string_data = path_leaf(data_image)
    string_label = 'haze' + string_data[5:]
    # print(string_data)
    print(string_label)
    label_image_name = LABEL_PATH + string_label
    #BI_img_name = BI_PATH + HR_img_name[12:19] + '.png'
    # print(label_image_name)
    imgData = cv2.imread(data_image)
    imgLabel = cv2.imread(label_image_name)
    # normalizing the input and target images
    imgData_normalized = imgData/255.0
    imgLabel_normalized = imgLabel/255.0
    #cv2.imshow('image',imgLabel_normalized)
    #cv2.imshow('data',imgData_normalized)
    #cv2.waitKey(0)
    #input_Data = np.array([imgData_normalized])
    #input_Label = np.array([imgLabel_normalized])
    # structuring them for tensor flow
    #input_elem = np.rollaxis(input_Data, 0, 4)
    #target_elem = np.rollaxis(input_Label, 0, 4)
    #(hei, wid) = input_elem.shape[0:2]
    #subim_input = input_elem[:, :, :,0]
    #subim_target = target_elem[:, :, :,0]
    INPUT[p[k], :, :, :] = imgLabel_normalized
    TARGET[p[k], :, :, :] = imgData_normalized
    #INPUT[k+total, :, :, :] = imgData_normalized
    #TARGET[k+total, :, :, :] = imgLabel_normalized
    #cv2.imshow('image1',INPUT[p[k]])
    #cv2.imshow('data1',TARGET[p[k]])
    #cv2.waitKey(0)
    k = k + 1
    #if k==total:
    #    break
    
    #INPUT = np.append(INPUT, imgData_normalized[np.newaxis, ...], axis=0)
    #TARGET = np.append(TARGET, imgLabel_normalized[np.newaxis, ...], axis=0)
    #count = count + 1
    print(str(k) + '-INPUT' + str(INPUT.shape) + '-TARGET' + str(TARGET.shape))
    sys.stdout.flush() #?
    #time.sleep(.1)  #?
    
    #i += 1
    #creation of patches for individual images complete

# start creating a single h5 file by combining all files and shuffling them

#print('>>>Start shuffling Images:')
# function for shuffling images
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#INPUT, TARGET = unison_shuffled_copies(INPUT, TARGET)
dset_input = h5fw.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
INPUT = None
print('>>>>INPUT file generated')
dset_target = h5fw.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
print('>>>>TARGET file generated')
print('>>>>save file' + 'training' + 'INPUT_' + str(SIZE_INPUT) + 'TARGET_' + str(SIZE_TARGET))
h5fw.close()
#h5fw_b = h5py.File(str('low_75_BW_' + str(SIZE_INPUT) + '.h5'), 'w')
#dset_input = h5fw_b.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
#dset_target = h5fw_b.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
#h5fw_b.close()
