#%% set all packages
import os
from sys import platform
import shutil
from PIL import Image
import numpy as np
import glob
import random

#%% functions
def rgb2gray(rgb):

    rgb = np.array(rgb)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def check4white(gray, tresh_value = 0.5):

    mask = gray > 220
    if sum(sum(mask)) > tresh_value * (len(mask) * len(mask)):
        all_white = True
    else:
        all_white = False

    return (all_white)


#%% get the sds-path
project_dir = "/dieSchöneUndDasBiest_HE/"
file_name = "H.18.4262_nurHämalaun.vmic"

if platform == "linux":
    sds_path = '/home/mr38/sds_hd/sd18a006/Marlen/datasets/' + project_dir
elif platform == "win32":
    sds_path = '//lsdf02.urz.uni-heidelberg.de/sd19G003/Marlen/datasets/' + project_dir

src_dir = sds_path + '/tiles/' + file_name + '/tissue/'
nn_dir = sds_path + '/cycleGan/' + file_name

#%% Creating Train / Val / Test folders (One time use)
test_ratio = 0.15
val_ratio = 0.15

#%% iterate over it

#% prepare the directories
# train folder
train_dir = nn_dir + '/train/'
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)
# validation folder
val_dir = nn_dir + '/val/'
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)
# test folder
test_dir = nn_dir + '/test/'
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)

#% prepare the data
# Creating partitions of the data after shuffeling

allFileNames = glob.glob(src_dir + '/*.tif')
np.random.shuffle(allFileNames)
sz = len(allFileNames)

train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [round(sz * (1.0 - test_ratio - val_ratio)),
                                                           round(sz * (1 - test_ratio))])

train_FileNames = [name for name in train_FileNames.tolist()]
val_FileNames = [name for name in val_FileNames.tolist()]
test_FileNames = [name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

#% Copy-pasting images
for name in train_FileNames:
    filename_w_ext = os.path.basename(name)
    shutil.copy(name, train_dir + '/' + filename_w_ext)

for name in val_FileNames:
    filename_w_ext = os.path.basename(name)
    shutil.copy(name, test_dir + '/' + filename_w_ext)

for name in test_FileNames:
    filename_w_ext = os.path.basename(name)
    shutil.copy(name, val_dir + '/' + filename_w_ext)

#% done
print('DONE!')
