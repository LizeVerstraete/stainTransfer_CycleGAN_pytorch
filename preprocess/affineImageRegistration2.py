from pystackreg import StackReg
from skimage import io
from matplotlib import pyplot as plt
import cv2
from sys import platform
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.color import rgb2gray
from skimage.color import gray2rgb


# define the name of the reference and moving images
ref_img_name = 'H.18.4262_HE_normal_9.tif'
mov_img_name = 'H.18.4262_kurz_9.tif'

if __name__ == '__main__':

    # get sds directory
    if platform == "linux":
        sds_path = Path('/home/mr38/sds_hd/sd18a006')
        if not sds_path.exists():
            sds_path = Path('/home/marlen/sds_hd/sd18a006')
    elif platform == "win32":
        sds_path = Path('//lsdf02.urz.uni-heidelberg.de/sd18A006')
    else:
        print('error: path cannot be defined! Abort')
        exit(1)

    # add directory containing data
    sds_path /= 'Marlen/stainNormalization_cycleGAN\datasets\dieSchöneUndDasBiest_HE\exports'

    ref_dir = sds_path / ref_img_name
    mov_dir = sds_path / mov_img_name

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(ref_dir))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(mov_dir))
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage())

#####
    ref = io.imread(ref_dir)
    mov = io.imread(mov_dir)

    plt.figure()
    plt.imshow(ref)
    plt.show()

    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    plt.figure()
    plt.imshow(backtorgb)
    plt.show()

    # ref = Image.open(ref_dir)
    # mov = Image.open(mov_dir)

    # ref.show()
    # mov.show()

    # get image dimensions
    ref_width = ref.shape[0]
    ref_height = ref.shape[1]
    mov_width = mov.shape[0]
    mov_height = mov.shape[1]
    # ref_width = ref.size[0]
    # ref_height = ref.size[1]
    # mov_width = mov.size[0]
    # mov_height = mov.size[1]

    # zero padding (adjust images to same sizes)
    max_val = np.max(ref)
    col = [max_val, max_val, max_val] # background color (white)
    ref = cv2.copyMakeBorder(ref, max(0, mov_width - ref_width), 0,  max(0, mov_height - ref_height), 0, cv2.BORDER_CONSTANT)
    mov = cv2.copyMakeBorder(mov, max(0, ref_width - mov_width), 0,  max(0, ref_height - mov_height), 0, cv2.BORDER_CONSTANT)

    # plot images
    plt.figure()
    plt.imshow(ref, cmap=None)
    plt.title(str(ref_img_name) + ' (zeropadded)')
    plt.show()

    plt.figure()
    plt.imshow(mov, cmap=None)
    plt.title(str(mov_img_name) + ' (zeropadded)')
    plt.show()

    # reshape
    w = ref.shape[0]
    h = ref.shape[1]

    ref = ref.reshape(w,-1)
    mov = mov.reshape(w, -1)

    # Rigid body transformation
    sr = StackReg(StackReg.RIGID_BODY)
    # sr = StackReg(StackReg.SCALED_ROTATION)
    # ref_transformed_r = sr.register_transform(ref[:,:,0], mov[:,:,0])
    # ref_transformed_g = sr.register_transform(ref[:,:,1], mov[:,:,1])
    # ref_transformed_b = sr.register_transform(ref[:,:,2], mov[:,:,2])
    # ref_transformed = np.dstack((ref_transformed_r, ref_transformed_g, ref_transformed_b))

    ref_transformed = sr.register_transform(ref, mov)

    ref_transformed = ref_transformed.reshape(w, h, 3)

    # plot transform
    plt.figure()
    plt.imshow(ref_transformed, cmap=None)
    plt.title('rigid transformation of ' + str(mov_img_name))
    plt.show()

    out_name = mov_img_name[:-4] +'_rigid_transf.tif'
    io.imsave(sds_path / out_name, ref_transformed)




