# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from torchvision import transforms,models
import torch.nn as nn
from PIL import Image
import clip
import pdb
import argparse
import glob
import h5py
import json
from tqdm import tqdm
#from parse_config import ConfigParser
import model.model_video_unfused as module_arch
from pathlib import Path
from collections import OrderedDict
from segment import cpd_auto

max_segment_num = 20
max_frame_num = 200


import warnings
warnings.filterwarnings("ignore")

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

config = read_json(Path('egoclip.json'))

class Image_Dataset(Dataset):
    def __init__(self, image_list, transform=None):

        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.image_list[idx]
        
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            sample = self.transform(image)
          
        return sample



def extract_video_features(args, video_num):
    
    '''
    parser = argparse.ArgumentParser(description='Extract CLIP features')
    #parser.add_argument('--model_name', default='ViT-B32', type=str, help='name of the model')
    #parser.add_argument('--root_dir', type=str, help='name of root feature dir')
    parser.add_argument('--cuda_base', help="in form cuda:x")
    parser.add_argument('--device_ids', help='delimited list input', type=lambda s: [int(item) for item in s.split(',')])
    #parser.add_argument('--h5_path', type=str, help="path to store the extracted features")
    #parser.add_argument('--config', default='egoclip.json', type=str, help='config file path (default: None)')
    #parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: None)')

    args = parser.parse_args()
    '''

    args.root_dir = '../Datasets/QFVS/data/video_frames/P0' + str(video_num) + '-moviepy/'

    image_list = sorted(filter(os.path.isfile,
                        glob.glob(os.path.join(args.root_dir,'*.jpg'))))


    transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    #config = ConfigParser(parser)
    #model = config.initialize('arch', module_arch)

    model = getattr(module_arch, config['arch']['type'])(config['arch']['args']['video_params'], config['arch']['args']['text_params'], 
                                 projection=config['arch']['args']["projection"], load_checkpoint=config['arch']['args']["load_checkpoint"])

    data = Image_Dataset(image_list, transform=transform)
    dataloader = DataLoader(data, batch_size=5, drop_last=True, shuffle=False, num_workers=4)

    device = torch.device(args.cuda_base if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = nn.DataParallel(model, device_ids = args.device_ids)

    feature = []

    with torch.no_grad():
        for i, image in enumerate(tqdm(dataloader)):
            image = image.to(device)
            b, c, h, w = image.shape
            image_reshape = image.reshape(int(b/5), 5, c, h, w)
            output = model(image_reshape)
            output = output.detach().to('cpu').numpy()
            feature.append(output)
            
        feature = np.concatenate(feature, axis=0)
        frame_num=feature.shape[0]
        print("Number of Frames: ", frame_num)

        K=feature[:,0,:]
        K=np.dot(K,K.T)

        cps,_=cpd_auto(K,max_segment_num-1,1,desc_rate=1,verbose=False,lmax=max_frame_num-1) #int(K.shape[0]/25)
        seg_num=len(cps)+1

        assert seg_num<=max_segment_num

        seg_points=cps
        seg_points=np.insert(seg_points,0,0)
        seg_points=np.append(seg_points,frame_num)

        segments=[]
        for i in range(seg_num):
            segments.append(np.arange(seg_points[i],seg_points[i+1],1,dtype=np.int32))

        assert len(segments)<=max_segment_num

        for seg in segments:
            assert len(seg)<=max_frame_num

        seg_len=np.zeros((max_segment_num),dtype=np.int32)
        for index,seg in enumerate(segments):
            seg_len[index]=len(seg)

        # features

        for seg in segments:
            for frame in seg:
                assert frame<frame_num

        feature_dim=768

        features=torch.zeros((max_segment_num, max_frame_num, feature.shape[1], feature_dim))
        for seg_index,seg in enumerate(segments):
            for frame_index,frame in enumerate(seg):
                features[seg_index,frame_index]=torch.tensor(feature[frame])
                # features[seg_index,frame_index]=F.avg_pool1d(t.tensor(feature[frame]).unsqueeze(0).unsqueeze(0),kernel_size=2,stride=2)
        
        
        print("Feature Extraction Done!")

        return features, seg_len

