""" Helper tool to convert animation from snaphosts of png images, using ImageMagick.
"""
import argparse
import glob
import os
import subprocess

import numpy as np


def extract_num(filename):
    basename = os.path.basename(filename)
    n = basename.split('.')[0].split('_')[-1]
    n = int(n)
    return n


def sort_by_filename(filenames):
    n = [extract_num(filename) for filename in filenames]
    idx = np.array(n).argsort()
    return [filenames[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir')
    parser.add_argument('savename')
    args = parser.parse_args()

    png_files = glob.glob(os.path.join(args.target_dir, '*.png'))
    assert len(png_files) > 0

    png_files = sort_by_filename(png_files)

    cmd = ['convert'] + png_files + [args.savename]
    try:
        subprocess.check_output(cmd)
    except:
        print(f"Error in {cmd}")


if __name__ == '__main__':
    main()
