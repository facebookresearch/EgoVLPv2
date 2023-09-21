# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import pandas as pd
import pdb
from base.base_dataset import TextVideoDataset
# from data_loader.transforms import init_transform_dict, init_video_transform_dict
from transforms import init_transform_dict, init_video_transform_dict
import json
import transformers


class EgoTaskQA(TextVideoDataset):
    def _load_metadata(self, args):
        metadata_dir = 'Data/qa/' + args.dataset_split_type
        split_files = {
            'train': 'formatted_train_qas_encode.json',
            'val': 'formatted_val_qas_encode.json',            # there is no test
            'test': 'formatted_test_qas_encode.json',  # there is no test
        }
        target_split_fp = split_files[self.split]

        with open(os.path.join(metadata_dir, target_split_fp),'r') as load_f:
            metadata = json.load(load_f)
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = pd.DataFrame(metadata)

        if self.split in ['train']:
            self.frame_sample = 'rand'
        elif self.split in ['val', 'test']:
            self.frame_sample = 'uniform'

    def _get_video_path(self, sample):
        rel_video_fp = sample['interval'] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp

    def _get_video_frames(self, video_fp):
        video_loading = self.video_params.get('loading', 'strict')
        # pdb.set_trace()
        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'])
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'], self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp  = self._get_video_path(sample)

        final = self._get_video_frames(video_fp)

        #pdb.set_trace()
        meta_arr = {'type': sample['type'], 'category': sample['category'], 'semantic': sample['semantic'], 'reasoning': sample['reasoning_type'].split('$')}
        return {'video': final, 'text': sample['question'], 'meta': meta_arr, 'answer': sample['answer_encode']}

def collate_func(batch):
    
    config = 'configs/egotaskqa.json'
    unique_dict = torch.load('reasoning_unique_cat.pth')
    f = open(config)
    config = json.load(f)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])
    video_frames_list, text_tokens_list, attention_masks_list, answers_list, reasoning_list = [], [], [], [], []
                            
    for i, item in enumerate(batch):
        video_frames_list.append(item['video'])
        tokenized_text = tokenizer(item['text'].strip().lower(), return_tensors='pt', padding='max_length', max_length=30, truncation=True)
        text_tokens_list.append(tokenized_text['input_ids'])
        attention_masks_list.append(tokenized_text['attention_mask'])
        answers_list.append(torch.tensor(item['answer']))

        index_list = []
        for _idx in range(len(item['meta']['reasoning'])):
            index_list.append(unique_dict[item['meta']['reasoning'][_idx]])
        reasoning_list.append(torch.tensor(index_list))

    video_frames = torch.stack(video_frames_list, dim=0)
    text_tokens = torch.stack(text_tokens_list, dim=0).squeeze(1)
    attention_masks = torch.stack(attention_masks_list, dim=0).squeeze(1)
    answers = torch.stack(answers_list, dim=0)

    # # torch.Size([4, 20, 3, 224, 224]) torch.Size([4, 12]) torch.Size([4]) torch.Size([4, 20, 2048]) torch.Size([4, 25, 15])
    return video_frames, text_tokens, attention_masks, answers, reasoning_list

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoTaskQA",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="EgoTaskQA/qa_videos",
        meta_dir="Data/qa/indirect",
        tsfms=init_video_transform_dict()['test'],
        reader='decord',
        split='train',
        neg_param=60
    )
    dataset = EgoTaskQA(**kwargs)
    import tqdm
    max_class = 0
    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]
        pdb.set_trace()
        class_no = item['answer']
        if class_no > max_class:
            max_class = class_no
        # print(item.keys())
    print(max_class)
