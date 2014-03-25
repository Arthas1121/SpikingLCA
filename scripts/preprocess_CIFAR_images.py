#!/usr/bin/env python

"""
Author: Miha Pelko, mpelko@gmail.com
Date: 25.2.2014

Converts all the color images in a target folder to bw images.
"""


import sys
from LCA.LCA_common import rgb_to_grey, plottable_rgb_matrix
import neurovivo.common as cmn
import png

def main():
    usage = "Usage: preprocess_CIFAR_images.py <data_source_file_path>  <location_name_prefix> "+\
    "<first_int>"

    assert len(sys.argv) == 4, usage
    
    SOURCE_FILE = sys.argv[1]
    LOCATION_PREFIX = sys.argv[2]
    FIRST_INT = int(sys.argv[3])

    db = cmn.load_pickle(SOURCE_FILE)

    for i, color_img_vec in enumerate(db["data"]):
        grey_img_vec = rgb_to_grey(color_img_vec)
        f = open('{}-{}.png'.format(LOCATION_PREFIX, FIRST_INT+i), 'wb')
        w = png.Writer(32, 32, greyscale=True, bitdepth=8)
        w.write(f, grey_img_vec.reshape(32,32))
        f.close()

if __name__ == "__main__":
    main()

