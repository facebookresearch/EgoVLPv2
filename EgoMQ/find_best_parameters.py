# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import argparse

parser = argparse.ArgumentParser(description='Find Best HyperParameters')
parser.add_argument('--root_folder', type=str, help='Root output folder')

args = parser.parse_args()

file_list = glob.glob(str(args.root_folder) + '/*.txt')

best_map = 0

for txt_file in file_list:
    file1 = open(txt_file, 'r')
    Lines = file1.readlines()

    for line in Lines:
        if 'Average-mAP:' in line.strip():
            if float(line.split( )[-1]) > best_map:
                best_map = float(line.split( )[-1])
                best_file_name = txt_file

print("File with the best results: ", best_file_name)
