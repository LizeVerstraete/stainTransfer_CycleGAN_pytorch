import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    dir_name = "test_paired"
    input_dir = "//lsdf02.urz.uni-heidelberg.de/sd18A006/Marlen/datasets/HE_H.18.4262/512/" + dir_name
    output_dir = input_dir + "/histo/"
    assure_path_exists(output_dir)

    for filename in os.listdir(input_dir):
        path = os.path.join(input_dir, filename)
        if os.path.isdir(path):
            # skip directories
            continue
        img = cv2.imread(os.path.join(input_dir,filename))
        depth = img.shape[2]
        for z in range(depth):
            im = img[:,:,z]
            mi = im.min()
            ma = im.max()
            if mi < 0 or ma > 255:
                print("range error: min=" + str(mi) + " max=" + ma )
                exit()

            # V1
            # plt.hist(im.ravel(), 256, [0, 256])

            # V2
            # calculate mean value from RGB channels and flatten to 1D array
            vals = im.flatten()
            # plot histogram with 255 bins
            # b, bins, patches = plt.hist(vals, 255, stacked=True, density=True)

            counts, bins = np.histogram(vals, 255)
            counts = (counts - min(counts)) / (max(counts) - min(counts))
            plt.hist(bins[:-1], bins, weights=counts)

            plt.xlim([0, 255])
            # plt.show()
            #
        plt.title(dir_name)
        # plt.show()
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()