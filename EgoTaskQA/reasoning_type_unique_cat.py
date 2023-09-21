# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import numpy as np
import pandas as pd
import pdb
import torch

def unique_cat(json_file_path):

    f = open(json_file_path)
    data = json.load(f)
    metadata = pd.DataFrame(data)

    l = []

    for idx in range(len(metadata)):
        l.append(metadata.iloc[idx]['reasoning_type'].split('$'))

    print(max([len(lst) for lst in l]))

    flat_list = [item for sublist in l for item in sublist]
    unique_list = np.unique(flat_list)

    return unique_list.tolist()


indirect_train = unique_cat('Data/qa/indirect/formatted_train_qas_encode.json')
indirect_test = unique_cat('Data/qa/indirect/formatted_test_qas_encode.json')
indirect_val = unique_cat('Data/qa/indirect/formatted_val_qas_encode.json')

direct_train = unique_cat('Data/qa/direct/formatted_train_qas_encode.json')
direct_test = unique_cat('Data/qa/direct/formatted_test_qas_encode.json')
direct_val = unique_cat('Data/qa/direct/formatted_val_qas_encode.json')

concat_unique_list = np.unique(list(indirect_train + indirect_test + indirect_val + direct_train + direct_test + direct_val)) 

unique_dict = {}
index = 0

for k in concat_unique_list:
    unique_dict.update({k: int(index)})
    index += 1

