# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import h5py
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
import pdb
import json
from collections import OrderedDict
from tqdm import tqdm

from dataset_prompt import UCTDataset
from utils_QFVS import load_pickle
import math
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path
import model.model_text_unfused as module_arch_text
import model.model_fused as module_arch_fused


class Extract_Multimodal_Features():
    def __init__(self,config, last_bias, train_videos, test_video, cuda_base, device_ids, features):
        self.config=config
        self.last_bias=last_bias
        self.train_videos = train_videos
        self.test_video = test_video
        self.cuda_base = cuda_base
        self.device_ids = device_ids
        self.features = features
        self._build_dataloader()
        self._bulid_model()

    def read_json(self, fname):
        with fname.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)


    def _bulid_model(self):

        config_egoclip = self.read_json(Path('qfvs.json'))

        self.model_text_unfused = getattr(module_arch_text, config_egoclip['arch']['type'])(config_egoclip['arch']['args']['video_params'], config_egoclip['arch']['args']['text_params'],
                                projection=config_egoclip['arch']['args']["projection"], load_checkpoint=config_egoclip['arch']['args']["load_checkpoint"])

        self.model_fused = getattr(module_arch_fused, config_egoclip['arch']['type'])(config_egoclip['arch']['args']['video_params'], config_egoclip['arch']['args']['text_params'],
                                projection=config_egoclip['arch']['args']["projection"], load_checkpoint=config_egoclip['arch']['args']["load_checkpoint"])
        
        self.config_egoclip = config_egoclip

    def _build_dataset(self):
        
        return UCTDataset(self.config, self.train_videos + [self.test_video], self.features)

    def _build_dataloader(self):
        dataset=self._build_dataset()
        self.dataloader=DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(len(self.dataloader))
    

    def extract(self):

        device = torch.device(self.cuda_base if torch.cuda.is_available() else 'cpu')
        criterion=torch.nn.BCEWithLogitsLoss()
        
        self.model_fused = self.model_fused.to(device)
        self.model_text_unfused = self.model_text_unfused.to(device)
        self.model_fused = nn.DataParallel(self.model_fused, device_ids = self.device_ids)
        self.model_text_unfused = nn.DataParallel(self.model_text_unfused, device_ids = self.device_ids)
        
        self.model_fused.eval()
        self.model_text_unfused.eval()

        features_dict = {'1':[], '2':[], '3':[], '4':[]}

        for epoch in range(1):

            step = epoch*len(self.dataloader)
            batch_count=0
            
            for _idx, (features, seg_len, concept1_token, concept1_mask, concept2_token, concept2_mask, query_token, query_mask, concept1_GT, concept2_GT, mask_GT, oracle_summary, video_id) in enumerate(tqdm(self.dataloader)):
                
                
                train_num=seg_len.shape[0]
                batch_count+=1

                mask=torch.zeros(train_num,self.config["max_segment_num"],self.config["max_frame_num"],dtype=torch.bool).cuda()
                for i in range(train_num):
                    for j in range(len(seg_len[i])):
                        for k in range(seg_len[i][j]):
                            mask[i][j][k]=1

                
                fusedfeat_concept1_batched_list = []
                fusedfeat_concept2_batched_list = []
                fusedfeat_oracle_batched_list = []


                with torch.no_grad():
                    
                    concept1_embed, extend_text_mask_concept1 = self.model_text_unfused(concept1_token, concept1_mask)
                    concept2_embed, extend_text_mask_concept2 = self.model_text_unfused(concept2_token, concept2_mask)
                    query_embed, extend_text_mask_query = self.model_text_unfused(query_token, query_mask)

                    concept1_embed = concept1_embed.unsqueeze(1).repeat(1,200,1,1)
                    concept2_embed = concept2_embed.unsqueeze(1).repeat(1,200,1,1)
                    query_embed = query_embed.unsqueeze(1).repeat(1,200,1,1)
                    extend_text_mask_concept1 = extend_text_mask_concept1.unsqueeze(1).repeat(1,200,1,1,1)
                    extend_text_mask_concept2 = extend_text_mask_concept2.unsqueeze(1).repeat(1,200,1,1,1)
                    extend_text_mask_query = extend_text_mask_query.unsqueeze(1).repeat(1,200,1,1,1)

                    for j in range(concept1_embed.size(0)):

                        fusedfeat_concept1_list = []
                        fusedfeat_concept2_list = []
                        fusedfeat_oracle_list = []
                        
                        for idx in range(features.size(1)):

                            concept1_fusedfeat = self.model_fused(features[j, idx], concept1_embed[j], extend_text_mask_concept1[j])
                            concept2_fusedfeat = self.model_fused(features[j, idx], concept2_embed[j], extend_text_mask_concept2[j])
                            oracle_fusedfeat = self.model_fused(features[j, idx], query_embed[j], extend_text_mask_query[j])

                            fusedfeat_concept1_list.append(concept1_fusedfeat)
                            fusedfeat_concept2_list.append(concept2_fusedfeat)
                            fusedfeat_oracle_list.append(oracle_fusedfeat)

                        fusedfeat_concept1_list = torch.stack(fusedfeat_concept1_list, 0)
                        fusedfeat_concept1_batched_list.append(fusedfeat_concept1_list)
                        fusedfeat_concept2_list = torch.stack(fusedfeat_concept2_list, 0)
                        fusedfeat_concept2_batched_list.append(fusedfeat_concept2_list)
                        fusedfeat_oracle_list = torch.stack(fusedfeat_oracle_list, 0)
                        fusedfeat_oracle_batched_list.append(fusedfeat_oracle_list)

                    fusedfeat_concept1_batched_list = torch.stack(fusedfeat_concept1_batched_list, 0)
                    fusedfeat_concept2_batched_list = torch.stack(fusedfeat_concept2_batched_list, 0)
                    fusedfeat_oracle_batched_list = torch.stack(fusedfeat_oracle_batched_list, 0)

                features_dict[str(video_id[0])].append({'seg_len':seg_len.squeeze(0), 'fusedfeat_concept1_batched_list':fusedfeat_concept1_batched_list.squeeze(0), 'fusedfeat_concept2_batched_list':fusedfeat_concept2_batched_list.squeeze(0), 'fusedfeat_oracle_batched_list':fusedfeat_oracle_batched_list.squeeze(0), 'concept1_GT':concept1_GT.squeeze(0), 'concept2_GT':concept2_GT.squeeze(0), 'mask_GT':mask_GT.squeeze(0), 'oracle_summary':oracle_summary.squeeze(0)})
                

        path1 = self.config_egoclip['arch']['args']["load_checkpoint"].split('.')[0]

        torch.save(features_dict, './multimodal_features/' + str(path1) + '_features.pt')
    
