# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import pickle

def load_json(filename):
    with open(filename, encoding='utf8') as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def save_pickle(object,filename):
    with open(filename, 'wb') as f:
        pickle.dump(object,f)
