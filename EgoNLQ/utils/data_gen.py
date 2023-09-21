# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import codecs
import multiprocessing
import os
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
import pdb

from utils.data_util import (
    load_json,
    load_lines,
    load_pickle,
    save_pickle,
    time_to_index,
)
import transformers
from base.transforms import init_transform_dict, init_video_transform_dict
import decord
import torch

PAD, UNK = "<PAD>", "<UNK>"

# from utils.frozen_ego_minimum import parse_config


class EpisodicNLQProcessor:
    def __init__(self):
        super(EpisodicNLQProcessor, self).__init__()
        self.idx_counter = 0
        
        config = '/cis/home/shraman/works_meta_2022/NLQ/Ego4D_episodic_memory-main/NLQ/VSLNet_unified/configs/nlq.json'
        f = open(config)
        self.config = json.load(f)
        self.meta_dir = self.config['data_loader']['args']['meta_dir']
        self.data_dir = self.config['data_loader']['args']['data_dir']
        self.num_frame = self.config['data_loader']['args']['video_params']['num_frames']
        #self.args = args
        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config['arch']['args']['text_params']['model'])
        self.transforms = init_video_transform_dict()['test']

    def reset_idx_counter(self):
        self.idx_counter = 0


    def create_meta_dict(self, split):
        
        split_files = {
            'train': 'nlq_train.json',
            'val': 'nlq_val.json',            # there is no test
            'test': 'nlq_test_unannotated.json'
        }
        # target_split_fp = split_files[self.split]

        # for split in ['train', 'val']:
        target_split_fp = split_files[split]
        ann_file = os.path.join(self.meta_dir, target_split_fp)
        with open(ann_file) as f:
            anno_json = json.load(f)

        meta_dict = dict()

        # forward video frames and text
        for anno_video in anno_json["videos"]:
            for anno_clip in anno_video["clips"]:
                clip_times = float(anno_clip["video_start_sec"]), float(anno_clip["video_end_sec"])
                meta_dict.update({anno_clip["clip_uid"]: {"video_uid": anno_video["video_uid"], "video_start": clip_times[0], "video_end": clip_times[1]}})

        return meta_dict

    def process_data_tan(self, data, scope, args):
        results = []
        meta_dict_scope = self.create_meta_dict(scope)

        for vid, data_item in tqdm(data.items(), total=len(data), desc=f"process episodic nlq {scope}"):
            fps = float(data_item["fps"])
            duration = float(data_item["num_frames"]) / fps
            zipper = zip(
                data_item["timestamps"],
                data_item["exact_times"],
                data_item["sentences"],
                data_item["annotation_uids"],
                data_item["query_idx"],
            )
            for timestamp, exact_time, sentence, ann_uid, query_idx in zipper:
                start_time = max(0.0, float(timestamp[0]) / fps)
                end_time = min(float(timestamp[1]) / fps, duration)

                if self._predictor == "EgoVLP":
                    text = self.tokenizer(sentence.strip().lower(), return_tensors='pt', padding='max_length', max_length=args.query_max_len, truncation=True)
                    text['input_ids'] = text['input_ids'].repeat(256, 1) ## This 256 is hardcoded to be the same as b_s in main.py
                    text['attention_mask'] = text['attention_mask'].repeat(256, 1) ## This 256 is hardcoded to be the same as b_s in main.py
                
                else:
                    raise NotImplementedError("args.predictor should be EgoVLP")
                
                record = {
                    "sample_id": self.idx_counter,
                    "vid": str(vid),
                    "num_frame": self.num_frame,
                    "transforms": self.transforms,
                    "data_dir": self.data_dir,
                    "s_time": start_time,
                    "e_time": end_time,
                    "exact_s_time": exact_time[0],
                    "exact_e_time": exact_time[1],
                    "duration": duration,
                    "query": sentence.strip().lower(),
                    "annotation_uid": ann_uid,
                    "query_idx": query_idx,
                    "query_feats":text,
                    "meta_dict_scope": meta_dict_scope,
                }
                results.append(record)
                self.idx_counter += 1


        return results

    def convert(self, args):
        predictor = args.predictor
        self._predictor = predictor
        self.reset_idx_counter()
        
        # load raw data
        train_data = load_json(os.path.join('jsons', "train.json"))
        val_data = load_json(os.path.join('jsons', "val.json"))
        test_data = load_json(os.path.join('jsons', "test.json"))

        # process data
        train_set = self.process_data_tan(train_data, scope="train", args=args)
        val_set = self.process_data_tan(val_data, scope="val", args=args)
        test_set = self.process_data_tan(test_data, scope="test", args=args)
        return train_set, val_set, test_set


def sample_frames_clips(start, end, vlen, acc_samples):
    start = max(0, start)
    end = min(vlen, end)

    intervals = np.linspace(start=start, stop=end, num=int(acc_samples) + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges
                      ]
    return frame_idxs

def read_frames_decord_start_end(video_path, start, end, num_frames):
    video_reader = decord.VideoReader(video_path, num_threads=8)
    vlen = len(video_reader)
    frame_idxs = sample_frames_clips(start, end, vlen, num_frames + 1)
    video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs)

    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs

def _get_video_path(sample, data_dir):
    rel_video_fp = sample
    full_video_fp = os.path.join(data_dir, rel_video_fp + '.mp4')
    return full_video_fp, rel_video_fp

def visual_feature_sampling(visual_feature, max_num_clips):
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
            new_visual_feature.append(visual_feature[s_idx].numpy())
    new_visual_feature = np.asarray(new_visual_feature)
    #pdb.set_trace()
    return torch.from_numpy(new_visual_feature)
    
def _get_video_text_feats(meta_dict_scope, clip_uid, num_frames, args, transforms, data_dir):
    
    video_fp, _ = _get_video_path(meta_dict_scope[clip_uid]["video_uid"], data_dir)

    fps = 1.87
    try:
        imgs, idxs = read_frames_decord_start_end(video_fp, meta_dict_scope[clip_uid]["video_start"]*30, meta_dict_scope[clip_uid]["video_end"]*30,
                                               (meta_dict_scope[clip_uid]["video_end"]-meta_dict_scope[clip_uid]["video_start"]) * fps * num_frames)
    except:
        print(f"Warning: missing video file {video_fp}.")

    if transforms is not None:
        imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
        imgs = transforms(imgs)
        imgs = imgs.transpose(0, 1)  # recover

    f, c, h, w = imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3]
    imgs = imgs[:(f // num_frames * num_frames), ]
    imgs = imgs.reshape(-1, num_frames, c, h, w)

    #imgs = visual_feature_sampling(imgs, max_num_clips=args.max_pos_len) ## max_num_clips = args.max_pos_len

    return imgs


def dataset_gen_EgoVLP(data, vfeat_lens, tokenizer, max_pos_len, scope, args, num_workers=1):
    dataset = []
    if not os.path.exists('./saved_video_frames/' + str(scope)):
        os.mkdir('./saved_video_frames/' + str(scope))

    print("DATASET INITIALIZED!!!!!!!")
    for indx, record in tqdm(enumerate(data), desc="dataset_gen_EgoVLP"):
        vid = record["vid"]
        if vid not in vfeat_lens:
            continue
        s_ind, e_ind, _ = time_to_index(
            record["s_time"], record["e_time"], vfeat_lens[vid], record["duration"]
        )
        
        video_frame_path = './saved_video_frames/' + str(scope) + '/' + str(record["vid"]) + '_frames.pt'

        if 'video_frame_path' in record['meta_dict_scope'][record['vid']]:
            video_path = record['meta_dict_scope'][record['vid']]['video_frame_path']
        else:
            video = _get_video_text_feats(record["meta_dict_scope"], record["vid"], record["num_frame"], args, record["transforms"], record["data_dir"])
            record['meta_dict_scope'][record['vid']].update({'video_frame_path':video_frame_path})
            video_path = video_frame_path
            torch.save(video, video_frame_path)

        result = {
            "sample_id": record["sample_id"],
            "vid": record["vid"],
            "video_frame_path": video_path,
            "s_time": record["s_time"],
            "e_time": record["e_time"],
            "duration": record["duration"],
            "query": record["query"],
            "s_ind": int(s_ind),
            "e_ind": int(e_ind),
            "v_len": vfeat_lens[vid],
            "text_tokens": record["query_feats"],
            "annotation_uid": record["annotation_uid"],
            "query_idx": record["query_idx"],
        } 
        #pdb.set_trace()
        dataset.append(result)
        
        #if indx == 40:
        #   print(len(dataset))
        #   break

    return dataset


def gen_or_load_dataset(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    data_dir = os.path.join("data", "dataset", args.task)
    #feature_dir = os.path.join("data", "features", args.task, args.fv)
    #if args.suffix is None:
    #    save_path = os.path.join(
    #        args.save_dir,
    #        "_".join(
    #            [args.task, args.fv, str(args.max_pos_len)]
    #        )
    #        + ".pkl",
    #    )
    #else:
    #    save_path = os.path.join(
    #        args.save_dir,
    #        "_".join(
    #            [args.task, args.fv, str(args.max_pos_len), args.suffix]
    #        )
    #        + ".pkl",
    #    )
    
    #if os.path.exists(save_path): ### Use this block with care, preferable to compute text fetaures before running
    #    dataset = load_pickle(save_path)
    #    return dataset
    

    feat_len_path = os.path.join("feature_shapes.json")
    #emb_path = os.path.join("data", "features", "glove.840B.300d.txt")
    # load video feature length
    vfeat_lens = load_json(feat_len_path)
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(args.max_pos_len, vfeat_len)
    # load data
    processor = EpisodicNLQProcessor()

    train_data, val_data, test_data = processor.convert(args)
    # generate dataset
    data_list = (
        [train_data, test_data]
        if val_data is None
        else [train_data, val_data, test_data]
    )
    
    if args.predictor == "EgoVLP":

        tokenizer = None
        train_set = dataset_gen_EgoVLP(
            train_data,
            vfeat_lens,
            tokenizer,
            args.max_pos_len,
            "train",
            args,
            num_workers=args.num_workers,
        )
        if val_data:
            val_set = dataset_gen_EgoVLP(
                val_data,
                vfeat_lens,
                tokenizer,
                args.max_pos_len,
                "val",
                args,
                num_workers=args.num_workers,
            )
        else:
            val_set = None
        test_set = dataset_gen_EgoVLP(
            test_data,
            vfeat_lens,
            tokenizer,
            args.max_pos_len,
            "test",
            args,
            num_workers=args.num_workers,
        )
        n_val = 0 if val_set is None else len(val_set)
        dataset = {
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
            "n_train": len(train_set),
            "n_val": n_val,
            "n_test": len(test_set),
        }
    else:
        raise NotImplementedError("args.predictor should be EgoVLP")

    #save_pickle(dataset, save_path)
    return dataset
