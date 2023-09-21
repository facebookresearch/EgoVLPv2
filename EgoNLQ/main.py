# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Main script to train/test models for Ego4D NLQ dataset.
"""
import argparse
import os
import pdb

import numpy as np
import options
import torch
import torch.nn as nn
from model.VSLNet import build_optimizer_and_scheduler, VSLNet
from tqdm.auto import tqdm
from utils.data_gen import gen_or_load_dataset, visual_feature_sampling
from utils.data_util import load_json, load_video_features, save_json
from utils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)
import json
import tqdm
from glob import glob
import torch
import torch.utils.data
from model.model import FrozenInTime
from utils.data_loader import get_test_loader, get_train_loader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(Dataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = torch.load(record["video_frame_path"])
        #video_feature = visual_feature_sampling(video_feature, max_num_clips=self.args.max_pos_len)
        text_tokens = record["text_tokens"]

        return video_feature, text_tokens, record["sample_id"], record["vid"], record["s_time"], record["e_time"], record["duration"], record["query"], record["s_ind"], record["e_ind"], record["v_len"], record["annotation_uid"], record["query_idx"]
        

    def __len__(self):
        return len(self.dataset)



def fused_feature_extract(args, dataset, feat_path, split='train_set'):
    
    train_set = Dataset(dataset = dataset[split])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=False,
    )

    f = open('configs/nlq.json')
    config = json.load(f)

    video_params = {"model": config['arch']['args']['video_params']['model'], "arch_config": config['arch']['args']['video_params']['arch_config'], "num_frames": config['arch']['args']['video_params']['num_frames'], "pretrained": True, "time_init":  config['arch']['args']['video_params']['time_init']}
    text_params = {"model": config['arch']['args']['text_params']['model'], "pretrained": True, "input": config['arch']['args']['text_params']['input']}
    projection_dim=config['arch']['args']["projection_dim"]
    load_checkpoint=args.model_name
    projection='minimal'
    load_temporal_fix='bilinear'
    task_names = 'EgoNCE_ITM_MLM'
    norm_layer = None
    embed_dim=768

    model = FrozenInTime(video_params, text_params, projection_dim=projection_dim, load_checkpoint=load_checkpoint,
                            projection=projection, load_temporal_fix=load_temporal_fix,
                            task_names = task_names, norm_layer = norm_layer, embed_dim=embed_dim)

    device = torch.device(args.cuda_base if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = nn.DataParallel(model, device_ids = args.device_ids)
    model.eval()

    feat_path = os.path.join(feat_path, split.split('_')[0])
    os.mkdir(feat_path)

    with torch.no_grad():
        for idx, batch in enumerate(train_loader):
        
            data = {"video": batch[0][0].to(device), "text": {"input_ids": batch[1]["input_ids"][0].to(device), "attention_mask": batch[1]["attention_mask"][0].to(device)}}
            #fused_video_features = model.forward(data)
            #fused_video_features = model.module.compute_video(data['video'])

            projection_dim = 768
            fused_video_features = torch.zeros(data['video'].shape[0], projection_dim)
            b_s = 256 ## Batch size inside the next loop, as (897, 4, 3, 224, 224) goes out of memory
            
            if data['video'].shape[0] % b_s == 0:
                times = data['video'].shape[0] // b_s 
            else:
                times = 1 + data['video'].shape[0] // b_s  ##  '1 +' is added by us
            
            for j in range(times):
                start = j*b_s
                if (j+1) * b_s > data['video'].shape[0]:
                    end = data['video'].shape[0]
                else:
                    end = (j+1)*b_s

                video_length = end - start

                #fused_video_features[start:end,] = model.module.compute_video(data['video'][start:end,])
                data_batch = {"video": data['video'][start:end,], "text": {"input_ids": data['text']['input_ids'][0:video_length,], "attention_mask": data['text']['attention_mask'][0:video_length,]}}
                fused_video_features[start:end,] = model.forward(data_batch)
            

            #fused_video_features = visual_feature_sampling(fused_video_features, max_num_clips=args.max_pos_len)
            
            dual_text_features = model.module.compute_text_tokens({"input_ids": batch[1]["input_ids"][0][0:1].to(device), "attention_mask": batch[1]["attention_mask"][0][0:1].to(device)}, is_proj=False)

            #saved_feature_dict = {"fused_video_features": fused_video_features.detach().cpu(), "dual_text_features": dual_text_features.detach().cpu(),
            #                      "sample_id": batch[2][0], "vid": batch[3][0], "s_time": batch[4][0], "e_time": batch[5][0],
            #                      "duration": batch[6][0], "query": batch[7][0], "s_ind": batch[8][0], "e_ind": batch[9][0],
            #                      "v_len": batch[10][0], "annotation_uid": batch[11][0], "query_idx": batch[12][0], "text_token": {"input_ids": batch[1]["input_ids"][0][0:1].to(device), "attention_mask": batch[1]["attention_mask"][0][0:1].to(device)}}  

            saved_feature_dict = {"fused_video_features": fused_video_features.detach().cpu(), "dual_text_features": dual_text_features.detach().cpu(),
                                  "sample_id": batch[2][0], "vid": batch[3][0], "s_time": batch[4][0], "e_time": batch[5][0],
                                  "duration": batch[6][0], "query": batch[7][0], "s_ind": batch[8][0], "e_ind": batch[9][0],
                                  "v_len": batch[10][0], "annotation_uid": batch[11][0], "query_idx": batch[12][0], "text_token": {"attention_mask": batch[1]["attention_mask"][0][0:1]}}

            torch.save(saved_feature_dict, os.path.join(feat_path, 'feat_' + str(idx) + '.pt'))


def main(args, parser):

    # set tensorflow configs
    # set_th_config(args.seed)

    feat_path = os.path.join(str(args.feature_dir), str(args.model_name).split('/')[-1].split('.')[0])

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)
        print("Feature is being extracted")
        # prepare or load dataset
        dataset = gen_or_load_dataset(args)
        args.char_size = dataset.get("n_chars", -1)
        args.word_size = dataset.get("n_words", -1)
        fused_feature_extract(args, dataset, feat_path, split='train_set')
        fused_feature_extract(args, dataset, feat_path, split='val_set')
        fused_feature_extract(args, dataset, feat_path, split='test_set')


    train_loader = get_train_loader(
        glob(os.path.join(feat_path, "train") + "/*.pt"), args=args
    )
    train_eval_loader = get_test_loader(
        glob(os.path.join(feat_path, "train") + "/*.pt"), args=args
    )
    val_loader = (get_test_loader(glob(os.path.join(feat_path, "val") + "/*.pt"), args=args))

    test_loader = get_test_loader(
        glob(os.path.join(feat_path, "train") + "/*.pt"), args=args
    )

    args.num_train_steps = len(train_loader) * args.epochs
    num_train_batches = len(train_loader)
    num_val_batches = 0 if val_loader is None else len(val_loader)
    num_test_batches = len(test_loader)

    # Device configuration
    device = torch.device(args.cuda_base if torch.cuda.is_available() else 'cpu')

    # create model dir
    #home_dir = os.path.join(
    #    args.model_dir,
    #    "_".join(
    #        [
    #           args.model_name,
    #            args.task,
    #            args.fv,
    #            str(args.max_pos_len),
    #            args.predictor,
    #        ]
    #    ),
    #)
    #if args.suffix is not None:
    #    home_dir = home_dir + "_" + args.suffix
    #model_dir = os.path.join(home_dir, "model")

    # train and test

    if args.mode.lower() == "train":
        eval_period = num_train_batches // 2
        #pdb.set_trace()
        # build model
        model = VSLNet(configs=args).to(device)

        optimizer, scheduler = build_optimizer_and_scheduler(model, configs=args)
        # start training
        best_metric = -1.0

        new_save_dir = './saved_nlq_results/' + str(args.model_name).split('/')[-1].split('.')[0]
        if not os.path.exists(new_save_dir):
            os.mkdir(new_save_dir)
        score_writer = open(
            os.path.join(new_save_dir, "eval_results_" + str(args.max_pos_len) + "_" + str(args.batch_size) + "_" + str(args.init_lr) + ".txt"), mode="w", encoding="utf-8"
        )
        print("start training...", flush=True)
        
        global_step = 0
        best_r1_iou03 = 0

        for epoch in range(args.epochs):
            model.train()
            for _, data in enumerate(train_loader):
                global_step += 1
                (
                    _,
                    vfeats,
                    vfeat_lens,
                    dual_text_feature,
                    s_labels,
                    e_labels,
                    h_labels,
                    query_attention_mask
                ) = data
                # prepare features
                vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)

                #pdb.set_trace()
                dual_text_feature = dual_text_feature.to(device)
                s_labels, e_labels, h_labels = (
                    s_labels.to(device),
                    e_labels.to(device),
                    h_labels.to(device),
                )
                query_attention_mask = query_attention_mask.to(device)

                if args.predictor == 'EgoVLP':
                    # generate mask
                    query_mask = query_attention_mask    

                else:
                    raise NotImplementedError("predictor should be EgoVLP")
                    
                # generate mask
                video_mask = convert_length_to_mask(vfeat_lens).to(device)
                # compute logits
                h_score, start_logits, end_logits = model(
                    vfeats, video_mask, query_mask, dual_text_feature
                )

                # compute loss
                highlight_loss = model.compute_highlight_loss(
                    h_score, h_labels, video_mask
                )
                loc_loss = model.compute_loss(
                    start_logits, end_logits, s_labels, e_labels
                )
                total_loss = loc_loss + args.highlight_lambda * highlight_loss
                
                #print(total_loss)
                #total_loss = loc_loss
                # compute and apply gradients
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )  # clip gradient
                
                optimizer.step()
                scheduler.step()
     
                # evaluate
                if (
                    global_step % eval_period == 0
                    or global_step % num_train_batches == 0
                ):
                    model.eval()
                    print(
                        f"\nEpoch: {epoch + 1:2d} | Step: {global_step:5d}", flush=True
                    )
                    #result_save_path = os.path.join(
                    #    model_dir,
                    #    f"{args.model_name}_{epoch}_{global_step}_preds.json",
                    #)
                    # Evaluate on val, keep the top 3 checkpoints.
                    results, mIoU, score_str = eval_test(
                        model=model,
                        data_loader=val_loader,
                        device=device,
                        mode="val",
                        epoch=epoch + 1,
                        global_step=global_step,
                        gt_json_path=args.eval_gt_json,
                        configs=args,
                    )

                    if best_r1_iou03 < results[0][0]:
                        best_r1_iou03 = results[0][0]
                        best_score_str = score_str
                        epoch_step = "Best Epoch: " + str(epoch+1) + ", Best Step: " + str(global_step)

                    print(score_str, flush=True)
                    score_writer.write(score_str)
                    score_writer.flush()
                    # Recall@1, 0.3 IoU overlap --> best metric.
                    if results[0][0] >= best_metric:
                        best_metric = results[0][0]
                    #    torch.save(
                    #        model.state_dict(),
                    #        os.path.join(
                    #            model_dir,
                    #            "{}_{}.t7".format(configs.model_name, global_step),
                    #        ),
                    #    )
                    #    # only keep the top-3 model checkpoints
                    #    filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
                    model.train()

        print("=" * 50)
        print(epoch_step + "\n" + best_score_str, flush=True)
        score_writer.write(epoch_step + "\n" + best_score_str)
        score_writer.flush()
        score_writer.close()

    
    '''

    elif configs.mode.lower() == "test":
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        # load previous configs
        pre_configs = load_json(os.path.join(model_dir, "configs.json"))
        parser.set_defaults(**pre_configs)
        configs = parser.parse_args()
        # build model
        model = VSLNet(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)

        # get last checkpoint file
        filename = get_last_checkpoint(model_dir, suffix="t7")
        model.load_state_dict(torch.load(filename))
        model.eval()
        result_save_path = filename.replace(".t7", "_test_result.json")
        results, mIoU, score_str = eval_test(
            model=model,
            data_loader=test_loader,
            device=device,
            mode="test",
            result_save_path=result_save_path,
            configs=configs,
        )
        print(score_str, flush=True)
    '''

if __name__ == "__main__":
    configs, parser = options.read_command_line()
    main(configs, parser)
