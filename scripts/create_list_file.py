#!/usr/bin/env python

"""
Author: Miha Pelko, mpelko@gmail.com
Date: 25.2.2014

Creates a file listing the image filenames.
"""
import sys
from LCA.LCA_common import rgb_to_grey, plottable_rgb_matrix
import neurovivo.common as cmn
import png

def main():
    usage = "Usage: preprocess_CIFAR_images.py <prefix> <last_int> "+\
    "<output_file_name>"

    assert len(sys.argv) == 4, usage
    
    PREFIX = sys.argv[1]
    LAST_INT = int(sys.argv[2])
    OUTPUT_FILE_NAME = sys.argv[3]

    res = ""
    for i in xrange(LAST_INT+1):
        res += "{0}-{1}.png\n".format(PREFIX, str(i))

    with open(OUTPUT_FILE_NAME, "w") as text_file:
        text_file.write(res)

if __name__ == "__main__":
    main()

