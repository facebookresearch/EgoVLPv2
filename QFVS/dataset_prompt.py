# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import h5py
import torch
from torch.utils.data.dataset import Dataset
from utils_QFVS import load_pickle, load_json
from torch.utils.data import DataLoader
import transformers
import pdb

class UCTDataset(Dataset):
    def __init__(self, config, train_videos, all_features):
        self.config=config
        self.train_videos = train_videos
        self.all_features = all_features
        self.dataset=[]
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
        for video_id in self.train_videos:
            for _ , _, files in os.walk("./Datasets/QFVS/data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
                for file in files:
                    self.dataset.append(file[:file.find("_oracle.txt")]+"_"+str(video_id))

    def __getitem__(self,index):
        video_id=self.dataset[index].split('_')[2]
        features = self.all_features[video_id]['feature']
        seg_len = self.all_features[video_id]['seg_len']

        transfer={"Cupglass":"Glass",
                  "Musicalinstrument":"Instrument",
                  "Petsanimal":"Animal"}

        concept1,concept2=self.dataset[index].split('_')[0:2]

        concept1_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        concept2_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        with open("./Datasets/QFVS/data/origin_data/Dense_per_shot_tags/P0"+video_id+"/P0"+video_id+".txt","r") as f:
            lines=f.readlines()
            for index,line in enumerate(lines):
                concepts=line.strip().split(',')
                if concept1 in concepts:
                    concept1_GT[index]=1
                if concept2 in concepts:
                    concept2_GT[index]=1

        shot_num=seg_len.sum()
        mask_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"],dtype=torch.bool)
        for i in range(shot_num):
            mask_GT[i]=1


        oracle_summary = torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        GT_summary_shots = []
        with open("./Datasets/QFVS/data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)+"/"+str(concept1)+"_"+str(concept2)+"_"+"oracle.txt","r") as f:
            for line in f.readlines():
                GT_summary_shots.append(int(line.strip()))
        GT_summary_shots_modified = [x - 1 for x in GT_summary_shots]
        for element in GT_summary_shots_modified:
            oracle_summary[element] = 1

        if concept1 in transfer:
            concept1=transfer[concept1]
        if concept2 in transfer:
            concept2=transfer[concept2]


        concept1_prompt = "There is a " + str(concept1)
        concept2_prompt = "There is a " + str(concept2)
        query_prompt = "There is a " + str(concept1) + " and a " + str(concept2)
        
        concept1_embed = self.tokenizer(concept1_prompt, return_tensors='pt', padding='max_length', max_length=15, truncation=True)
        concept2_embed = self.tokenizer(concept2_prompt, return_tensors='pt', padding='max_length', max_length=15, truncation=True)
        query_embed = self.tokenizer(query_prompt, return_tensors='pt', padding='max_length', max_length=15, truncation=True)

        concept1_token, concept1_mask = concept1_embed['input_ids'], concept1_embed['attention_mask']
        concept2_token, concept2_mask = concept2_embed['input_ids'], concept2_embed['attention_mask']
        query_token, query_mask = query_embed['input_ids'], query_embed['attention_mask']

        return features, seg_len, concept1_token[0], concept1_mask[0], concept2_token[0], concept2_mask[0], query_token[0], query_mask[0], concept1_GT, concept2_GT, mask_GT, oracle_summary, video_id

    def __len__(self):
        return len(self.dataset)



if __name__=='__main__':

    config=load_json("./config/config.json")


