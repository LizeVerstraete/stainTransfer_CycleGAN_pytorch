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
project_dir = '/dieSchöneUndDasBiest_HE/'
file_dir = '/tiles/H.18.4262_nurHämalaun.vmic/'

if platform == "linux":
    sds_path = '/home/mr38/sds_hd/sd18a006/Marlen/datasets/' + project_dir + file_dir
elif platform == "win32":
    sds_path = '//lsdf02.urz.uni-heidelberg.de/sd19G003/Marlen/datasets/' + project_dir + file_dir


#%% seperate white space images from tissue images (in seperate directories)
allFileNames = glob.glob(sds_path + '/*.tif')
allFileNames = random.sample(allFileNames, len(allFileNames))

tissue_folder = sds_path + '/tissue/'
if not os.path.exists(tissue_folder):
    os.makedirs(tissue_folder)
ws_folder = sds_path + '/whiteSpace/'
if not os.path.exists(ws_folder):
    os.makedirs(ws_folder)

# now iterate over the images
n_tissue, n_ws = 0, 0
for ifile in allFileNames:
    filename_w_ext = os.path.basename(ifile)
    filename, file_extension = os.path.splitext(filename_w_ext)
    if os.path.isfile(ifile) == False:
        continue
    try: # bad style, I know
        img = Image.open(ifile)
        # img = img.resize([500,500])
    except:
        print('error override')
        continue

    all_white = check4white(rgb2gray(img), 0.8)

    if all_white:
        #if n_ws > n_max_tiles_per_class:
        #   continue
        img.save(ws_folder + filename_w_ext)
        os.remove(sds_path + filename_w_ext)
        print(filename_w_ext + ' added in folder "WS"')
        n_ws += 1
    else:
        #if n_lung > n_max_tiles_per_class :
        #    continue
        img.save(tissue_folder + filename_w_ext)
        os.remove(sds_path + filename_w_ext)
        print(filename_w_ext + ' added in folder "Lung"')
        n_tissue += 1

    #if n_lung > n_max_tiles_per_class  and n_ws > n_max_tiles_per_class:
    #    break

print('Finally n = ' + str(n_tissue) + ' lung images and n = ' + str(n_ws) + ' white space images were added. Total= ' + str(n_tissue+n_ws))

# #%% Creating Train / Val / Test folders (One time use)
# nn_dir = 'cycleGan'
# classes_dir = ['tissue',
#                '/whiteSpace']
# val_ratio = 0.15
# test_ratio = 0.15
#
# #%% iterate over it
# for cls in classes_dir:
#
#     #% counter section
#     print('folder ' + cls + ' started')
#
#     #% prepare the directories
#     # train folder
#     trainFolder = nn_dir +'/train' + cls
#     if os.path.exists(trainFolder):
#         shutil.rmtree(trainFolder)
#     os.makedirs(trainFolder)
#     # validation folder
#     valFolder = nn_dir +'/val' + cls
#     if os.path.exists(valFolder):
#         shutil.rmtree(valFolder)
#     os.makedirs(nn_dir +'/val' + cls)
#     # test folder
#     testFolder = root_dir +'/test' + cls
#     if os.path.exists(testFolder):
#         shutil.rmtree(testFolder)
#     os.makedirs(root_dir +'/test' + cls)
#
#     #% prepare the data
#     # Creating partitions of the data after shuffeling
#     src = root_dir + cls  # Folder to copy images from
#
#     allFileNames = glob.glob(src + '/*.tif')
#     np.random.shuffle(allFileNames)
#     train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
#                                                               [int(len(allFileNames) * (1 - val_ratio + test_ratio)),
#                                                                int(len(allFileNames) * (1 - test_ratio))])
#
#     train_FileNames = [name for name in train_FileNames.tolist()]
#     val_FileNames = [name for name in val_FileNames.tolist()]
#     test_FileNames = [name for name in test_FileNames.tolist()]
#
#     print('Total images: ', len(allFileNames))
#     print('Training: ', len(train_FileNames))
#     print('Validation: ', len(val_FileNames))
#     print('Testing: ', len(test_FileNames))
#
#     #% Copy-pasting images
#     for name in train_FileNames:
#         shutil.copy(name, root_dir + '/train' + cls)
#
#     for name in val_FileNames:
#         shutil.copy(name, root_dir + '/val' + cls)
#
#     for name in test_FileNames:
#         shutil.copy(name, root_dir + '/test' + cls)
#
#     #% counter section
#     print('folder ' + cls + ' finished')
