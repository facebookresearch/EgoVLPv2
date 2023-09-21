# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import h5py
import numpy as np
import torch
import transformers
from semantic_evaluation import calculate_semantic_matching, load_videos_tag
import pdb
import json
from collections import OrderedDict
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from utils_QFVS import load_pickle
import math
import torch.nn as nn
import torch.nn.init as init
from pathlib import Path
from model.model_summary import Transformer_Encoder


def weights_init(m):
    init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
    if m.bias is not None:
        init.constant_(m.bias, 0.1)


class Feature_Dataset(Dataset):
    def __init__(self, feature_list):

        self.feature_list = feature_list

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, idx):

        feature = self.feature_list[idx]

        return feature['seg_len'], feature['fusedfeat_concept1_batched_list'], feature['fusedfeat_concept2_batched_list'], feature['fusedfeat_oracle_batched_list'], feature['concept1_GT'], feature['concept2_GT'], feature['mask_GT'], feature['oracle_summary']


class Runner_summary_transformer():
    def __init__(self,config, last_bias, train_videos, test_video, cuda_base, device_ids, feature_dict):
        self.config=config
        self.last_bias=last_bias
        self.train_videos = train_videos
        self.test_video = [test_video] #convert int to list
        self.cuda_base = cuda_base
        self.device_ids = device_ids
        self.feature_dict = feature_dict
        self._build_dataloader()
        self._bulid_model()
        self._build_optimizer()
        self.max_f1=0
        self.max_p=0
        self.max_r=0

    def _bulid_model(self):

        self.model_summary = Transformer_Encoder(self.config['model_dim'], self.config['nhead'], self.config['num_layers'])

    def _build_dataset(self):

        self.feature_list = []
        for video_id in self.train_videos:
            self.feature_list += self.feature_dict[str(video_id)]
        
        return Feature_Dataset(self.feature_list)

    def _build_dataloader(self):
        dataset=self._build_dataset()
        self.dataloader=DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])
        print(len(self.dataloader))

    def _build_optimizer(self):
        param_weights = []
        param_biases = []
        for param in self.model_summary.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        self.optimizer = torch.optim.AdamW(parameters, lr=self.config["lr"], weight_decay=0.1)

    def output(self):
        print(" max_p = ",self.max_p," max_r = ",self.max_r," max_f1 = ",self.max_f1)

    def adjust_learning_rate(self, config, optimizer, loader, step):
        max_steps = config["epoch"] * len(loader)
        warmup_steps = 2 * len(loader)
        base_lr = 0.00001
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]['lr'] = lr * config["learning_rate_weights"]
        optimizer.param_groups[1]['lr'] = lr * config["learning_rate_biases"]


    def train(self):

        device = torch.device(self.cuda_base if torch.cuda.is_available() else 'cpu')
        criterion=torch.nn.BCEWithLogitsLoss()
        self.model_summary.summ_head.apply(weights_init)
        torch.nn.init.normal_(self.model_summary.summ_head.weight, mean=0.0, std=0.1)
        torch.nn.init.constant_(self.model_summary.summ_head.bias, self.last_bias)
        
        self.model_summary = self.model_summary.to(device)
        self.model_summary = nn.DataParallel(self.model_summary, device_ids = self.device_ids)
 
        self.model_summary.train()
        scaler = torch.cuda.amp.GradScaler()

        self.evaluate(self.test_video,self.config["top_percent"],0)

        for epoch in range(self.config["epoch"]):

            step = epoch*len(self.dataloader)
            batch_count=0
            
            for _, (seg_len, fusedfeat_concept1_batched_list, fusedfeat_concept2_batched_list, fusedfeat_oracle_batched_list, concept1_GT, concept2_GT, mask_GT, oracle_summary) in enumerate(tqdm(self.dataloader)):
                
                train_num=seg_len.shape[0]
                batch_count+=1

                self.optimizer.zero_grad()

                mask=torch.zeros(train_num,self.config["max_segment_num"],self.config["max_frame_num"],dtype=torch.bool).cuda()
                for i in range(train_num):
                    for j in range(len(seg_len[i])):
                        for k in range(seg_len[i][j]):
                            mask[i][j][k]=1


                with torch.cuda.amp.autocast():
                    
                    concept1_score = self.model_summary(fusedfeat_concept1_batched_list, seg_len)
                    concept2_score = self.model_summary(fusedfeat_concept2_batched_list, seg_len)
                    oracle_score = self.model_summary(fusedfeat_oracle_batched_list, seg_len)
                    
                    loss=torch.zeros(1).to(device)

                    for i in range(train_num):
                        concept1_score_tmp=concept1_score[i].masked_select(mask[i]).unsqueeze(0)
                        concept2_score_tmp=concept2_score[i].masked_select(mask[i]).unsqueeze(0)
                        concept1_GT_tmp=concept1_GT[i].masked_select(mask_GT[i]).unsqueeze(0)
                        concept2_GT_tmp=concept2_GT[i].masked_select(mask_GT[i]).unsqueeze(0)

                        score_tmp = oracle_score[i].masked_select(mask[i]).unsqueeze(0)
                        oracle_summary_sample = oracle_summary[i][0:score_tmp.shape[1]].unsqueeze(0)

                        loss_concept1=criterion(concept1_score_tmp,concept1_GT_tmp.to(device))
                        loss_concept2=criterion(concept2_score_tmp,concept2_GT_tmp.to(device))
                        loss_oracle=criterion(score_tmp, oracle_summary_sample.to(device))
                        loss+=loss_concept1+loss_concept2+loss_oracle

                self.adjust_learning_rate(self.config, self.optimizer, self.dataloader, step)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                step+=1


            self.evaluate(self.test_video,self.config["top_percent"],epoch)

    def evaluate(self,test_video,top_percent,epoch):

        f1_sum=0
        p_sum=0
        r_sum=0

        evaluation_num=0


        test_feature_list = []
        for video_id in test_video:
            test_feature_list += self.feature_dict[str(video_id)]

        test_dataset = Feature_Dataset(test_feature_list)
        test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        self.model_summary.eval()
        
        evaluation_num=len(test_dataloader)

        for _, (seg_len, fusedfeat_concept1_batched_list, fusedfeat_concept2_batched_list, fusedfeat_oracle_batched_list, concept1_GT, concept2_GT, mask_GT, oracle_summary) in enumerate(test_dataloader):

            summaries_GT = torch.nonzero(oracle_summary[0:1,:], as_tuple=True)
            summaries_GT = summaries_GT[1].numpy().tolist()

            mask=torch.zeros(1,self.config["max_segment_num"],self.config["max_frame_num"],dtype=torch.bool).cuda()
            for i in range(1):
                for j in range(len(seg_len[i])):
                    for k in range(seg_len[i][j]):
                        mask[i][j][k]=1

            score = self.model_summary(fusedfeat_oracle_batched_list, seg_len)

            score=score.masked_select(mask)

            _,top_index=score.topk(int(score.shape[0]*top_percent))


            video_shots_tag = load_videos_tag(mat_path="./Tags.mat")
            p, r, f1 = calculate_semantic_matching(list(top_index.cpu().numpy()), summaries_GT, video_shots_tag, video_id=video_id-1)

            f1_sum+=f1
            r_sum+=r
            p_sum+=p

        if f1_sum/evaluation_num>self.max_f1:
            self.max_f1=f1_sum/evaluation_num
            self.max_p=p_sum/evaluation_num
            self.max_r=r_sum/evaluation_num

        print("p = ",p_sum/evaluation_num," r = ",r_sum/evaluation_num," f1 = ",f1_sum/evaluation_num)


