# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import sys
import json
import pandas as pd
import numpy as np
import torch
import transformers
from tqdm import tqdm

from base.base_dataset import TextVideoDataset
from base.transforms import init_transform_dict, init_video_transform_dict


class NaturalLanguageQueries(TextVideoDataset):

    def __init__(self, dataset_name, text_params, video_params, config='/cis/home/shraman/works_meta_2022/NLQ/Ego4D_episodic_memory-main/NLQ/VSLNet_unified/configs/nlq.json', split='train', tsfms=None, reader='decord'):
        
        f = open(config)
        config = json.load(f)
        meta_dir = config['data_loader']['args']['meta_dir']
        data_dir = config['data_loader']['args']['data_dir']
        self.num_frame = config['data_loader']['args']['video_params']['num_frames']
        #self.args = args
        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])
        
        super().__init__(dataset_name=dataset_name, text_params=text_params, video_params=video_params, data_dir=data_dir, meta_dir=meta_dir, split=split, tsfms=tsfms, cut=None, subsample=1, sliding_window_stride=-1, reader=reader, neg_param=None)

    def _load_metadata(self):
        split_files = {
            'train': 'nlq_train.json',
            'val': 'nlq_val.json',            # there is no test
            'test': 'nlq_test_unannotated.json'
        }
        # target_split_fp = split_files[self.split]

        self.metadata = pd.DataFrame(columns=['video_uid', 'clip_uid',
                                              'video_start_sec', 'video_end_sec',
                                              'query'])

        # for split in ['train', 'val']:
        target_split_fp = split_files[split]
        ann_file = os.path.join(self.meta_dir, target_split_fp)
        with open(ann_file) as f:
            anno_json = json.load(f)


        # forward video frames and text
        for anno_video in anno_json["videos"]:
            for anno_clip in anno_video["clips"]:
                clip_times = float(anno_clip["video_start_sec"]), float(anno_clip["video_end_sec"])
                for anno in anno_clip['annotations']:
                    for query in anno["language_queries"]:
                        clip_duration = clip_times[1] - clip_times[0]
                        if 'query' not in query.keys():
                            continue
                        if query['query'] is None:
                            continue
                        new = pd.DataFrame({
                            'video_uid': anno_video['video_uid'],
                            'clip_uid': anno_clip['clip_uid'],
                            'video_start_sec': clip_times[0],
                            'video_end_sec': clip_times[1],
                            'query': query["query"],}, index=[1])
                        self.metadata = self.metadata.append(new, ignore_index=True)

        self.transforms = init_video_transform_dict()['test']

    def _get_video_path(self, sample):
        rel_video_fp = sample[0]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp + '.mp4')
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        caption = sample['query']
        return caption

    def visual_feature_sampling(self, visual_feature, max_num_clips):
        num_clips = visual_feature.shape[0]
        if num_clips <= max_num_clips:
            return visual_feature
        idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
        idxs = np.round(idxs).astype(np.int32)
        idxs[idxs > num_clips - 1] = num_clips - 1
        new_visual_feature = []
        for i in range(max_num_clips):
            s_idx, e_idx = idxs[i], idxs[i + 1]
            if s_idx < e_idx:
                new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx].numpy(), axis=0))
            else:
                new_visual_feature.append(visual_feature[s_idx])
        new_visual_feature = np.asarray(new_visual_feature)
        return torch.from_numpy(new_visual_feature)

    
    def _get_video_text_feats(self, item):
        sample = self.metadata.iloc [item]
        video_fp, rel_fp = self._get_video_path(sample)
       
        
        fps = 1.87
        try:
            imgs, idxs = self.video_reader(video_fp, sample[2]*30, sample[3]*30,
                                               (sample[3]-sample[2]) * fps * self.video_params['num_frames'])
        except:
            print(f"Warning: missing video file {video_fp}.")

        if self.transforms is not None:
            imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0, 1)  # recover

        f, c, h, w = imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3]
        imgs = imgs[:(f // self.num_frame * self.num_frame), ]
        imgs = imgs.reshape(-1, self.num_frame, c, h, w)

        imgs = self.visual_feature_sampling(imgs, max_num_clips=256) ## max_num_clips = args.max_pos_len
        
        text = self._get_caption(sample)
        text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        text['input_ids'] = text['input_ids'].repeat(imgs.shape[0], 1)
        text['attention_mask'] = text['attention_mask'].repeat(imgs.shape[0], 1)
        

        meta_arr = {'video_uid': sample[0], 'clip_uid': sample[1], 'video_start': sample[2], 'video_end': sample[3], 'data': video_fp, 'dataset': self.dataset_name}
        data = {'video': imgs, 'text': text, 'meta' : meta_arr}
        #data = {'meta' : meta_arr}
        
        return data

    def __getitem__(self, item):
        return self._get_video_text_feats(item)

if __name__ == "__main__":
    split = 'val'
    kwargs = dict(
        dataset_name="Ego4d_NLQ",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        tsfms=init_video_transform_dict()['test'],
        reader='decord_start_end',
        split=split,
    )
    dataset = NaturalLanguageQueries(**kwargs)
    print(len(dataset))
    
    test_dict = {}
    c = 0
    for i, _ in tqdm(enumerate(range(len(dataset)))):
        item = dataset[i]
        if item['meta']['clip_uid'] in test_dict:
            if test_dict[item['meta']['clip_uid']][0] != item['meta']['video_start'] or test_dict[item['meta']['clip_uid']][1] != item['meta']['video_end']:
                c += 1
        test_dict.update({item['meta']['clip_uid']: (item['meta']['video_start'], item['meta']['video_end'])})
    print(c)

