
from PIL import Image
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

W = [256]
H = W
X = [0]
Y = X
dir_name = "testB_tiff"
input_dir = "/home/mr38/sds_hd/sd18a006/Marlen/GAN/datasets/MITOS-ATYPIA-14/cyclegan/test/" + dir_name + "/"
output_dir = input_dir + "../" + dir_name + "_cropped/"
assure_path_exists(output_dir)

for filename in os.listdir(input_dir):
    im = Image.open(os.path.join(input_dir, filename))
    for w in W:
        for h in H:
            for x in X:
                for y in Y:
                    region = im.crop((x, y, x+w, y+h))
                    output_name = output_dir + filename[:-4] + "_x" + str(x) + "_y" + str(y) + "_w" + str(w) + "_h" + str(h) + ".jpg"
                    region.save(output_name)