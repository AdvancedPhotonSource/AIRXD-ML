import os
import imageio
import numpy as np
from glob import glob
from PIL import Image as IMAGE

def DimensionReduction(image, size, TA, angle=20):
    """ Cropped image based on the size from the center. """
    size //= 2
    row_min, row_max = np.min(TA_[0]), np.max(TA_[0])
    col_min, col_max = np.min(TA_[1]), np.max(TA_[1])
    row_mid, col_mid = (row_max-row_min)//2, (col_max-col_min)//2
    row_min_, row_max_ = row_mid-size, row_mid+size
    col_min_, col_max_ = col_mid-size, col_mid+size
    assert (row_min_ > 0) or (col_min_ > 0), "The size is out of range, please reduce the size."
    return Image_[row_min_:row_max_+1, col_min_:col_max_+1]

def parse_imctrl(filename):
    controls = {'size': [2880, 2880], 'pixelSize': [150.0, 150.0]}
    keys = ['IOtth', 'PolaVal', 'azmthOff', 'rotation', 'distance', 'center', 'tilt', 'DetDepth']
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ln = line.split(':')
            if ln[0] in keys:
                if ln[1][0] == '[':
                    temp = []
                    temp_list = ln[1].split(',')
                    temp.append(float(temp_list[0][1:]))
                    try:
                        temp.append(float(temp_list[1][:-2]))
                    except:
                        temp.append(False)
                    controls[ln[0]] = temp
                else:
                    controls[ln[0]] = float(ln[1])

    return controls

class Dataset:
    def __init__(self):
        self.X = None
        self.y = None

    def get_images(self, path, ext='.tif'):
        ipath = sorted(glob(os.path.join(path, f'*{ext}')))
        #if path[-1] == '/':
        #    ipath = sorted(glob(path+f'*{ext}'))
        #else:
        #    ipath = sorted(glob(path+f'/*{ext}'))
        
        images = np.zeros((len(ipath), 2880, 2880))
        for i, ip in enumerate(ipath):
            images[i] += imageio.volread(ip)

        return images
