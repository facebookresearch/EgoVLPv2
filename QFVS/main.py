# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import yaml
import pdb
import json
import argparse 
import torch
import numpy as np
from runner_train import Runner_summary_transformer
from extract_features import extract_video_features
from extract_multimodal_features import Extract_Multimodal_Features
from pathlib import Path
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_videos', help='delimited list input', type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('--test_video', help='test video id', type=int)
parser.add_argument('--cuda_base', help="in form cuda:x")
parser.add_argument('--device_ids', help='delimited list input', type=lambda s: [int(item) for item in s.split(',')])

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

config_qfvs = read_json(Path('qfvs.json'))

args = parser.parse_args()
with open('./QFVS.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

path1 = config_qfvs['arch']['args']["load_checkpoint"].split('.')[0]

try:
    feature_dict = torch.load('./multimodal_features/' + str(path1) + '_features.pt')
except:
    features={}
    all_videos = args.train_videos + [args.test_video]
    for video_id in all_videos:
        feature, seg_len = extract_video_features(args, video_id)
        features.update({str(video_id) : {'feature': feature, 'seg_len': seg_len}})
    extract_multimodal_features = Extract_Multimodal_Features(config=config, last_bias=-1.91, train_videos=args.train_videos, test_video=args.test_video, cuda_base=args.cuda_base, device_ids=args.device_ids, features=features)
    extract_multimodal_features.extract()
    feature_dict = torch.load('./multimodal_features/' + str(path1) + '_features.pt')
    print("In Except")
    print("=================")

runner = Runner_summary_transformer(config=config, last_bias=-1.91, train_videos=args.train_videos, test_video=args.test_video, cuda_base=args.cuda_base, device_ids=args.device_ids, feature_dict=feature_dict)

runner.train()
runner.output()

