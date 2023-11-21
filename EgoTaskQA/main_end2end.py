# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys, os
from tqdm import tqdm
from datetime import datetime
import argparse, time
import pdb
from yaml import parse
from tqdm import tqdm
import json
import glob
import copy
import math
from model.video_qa_model_linear_end2end import FrozenInTime
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup)

from EgoTaskQA_dataset import EgoTaskQA, collate_func
from transforms import init_transform_dict, init_video_transform_dict
import torch.distributed as dist
from utils.util import state_dict_data_parallel_fix, ReasongingTypeAccCalculator


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='outputs_end2end/',
                        help='where to store ckpts and logs')

    parser.add_argument("--per_gpu_batch_size", type=int, default=32)

    parser.add_argument("--nepoch", type=int, default=36,  
                        help='num of total epoches')
    parser.add_argument("--lr", type=float, default=5e-5,  
                        help='')
    
    parser.add_argument("--i_val",   type=int, default=20, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test",   type=int, default=20, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=5, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=20, 
                        help='frequency of weight ckpt saving')
    
    parser.add_argument('--dataset_split_type', type=str, default='direct', help='direct/indirect')
    parser.add_argument('--num_frames_per_video', type=int, default=4)
    parser.add_argument('--frame_resolution', type=int, default=224)
    parser.add_argument('--model_name', type=str, help="Path of the pretrained checkpoint")
    parser.add_argument('--resume_finetune_model_path', type=str, default=None, help ="Path of the previously fine-tuned checkpoint for resume")
    parser.add_argument('--test_only_model_path', type=str, default=None, help ="Path of the final fine-tuned checkpoint only for evaluation")
    parser.add_argument('--base_data_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def main_worker(gpu, args):
    
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    dataset_train = EgoTaskQA(dataset_name="EgoTaskQA", text_params={"input": "text"},
                                  video_params={"input_res": args.frame_resolution,
                                                "num_frames": args.num_frames_per_video,
                                                "loading": "lax"},
                                  data_dir="./qa_videos",
                                  meta_dir="./Data/qa/" + args.dataset_split_type,
                                  tsfms=init_video_transform_dict()['test'], reader='decord', split='train', neg_param=60, args=args)
    
    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.per_gpu_batch_size, drop_last=True, sampler=sampler_train, pin_memory=True, collate_fn=collate_func)


    dataset_val = EgoTaskQA(dataset_name="EgoTaskQA", text_params={"input": "text"},
                                video_params={"input_res": args.frame_resolution,
                                              "num_frames": args.num_frames_per_video,
                                              "loading": "lax"},
                                data_dir="./qa_videos",
                                meta_dir="./Data/qa/" + args.dataset_split_type,
                                tsfms=init_video_transform_dict()['test'], reader='decord', split='val', neg_param=60, args=args)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=args.per_gpu_batch_size, drop_last=False, sampler=sampler_val, collate_fn=collate_func)


    dataset_test = EgoTaskQA(dataset_name="EgoTaskQA", text_params={"input": "text"},
                                 video_params={"input_res": args.frame_resolution,
                                               "num_frames": args.num_frames_per_video,
                                               "loading": "lax"},
                                 data_dir="./qa_videos",
                                 meta_dir="./Data/qa/" + args.dataset_split_type,
                                 tsfms=init_video_transform_dict()['test'], reader='decord', split='test', neg_param=60, args=args)
    sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.per_gpu_batch_size, drop_last=False, sampler=sampler_test, collate_fn=collate_func)

    f = open('configs/egotaskqa.json')
    config = json.load(f)

    video_params = {"model": config['arch']['args']['video_params']['model'], "arch_config": config['arch']['args']['video_params']['arch_config'], "num_frames": config['arch']['args']['video_params']['num_frames'], "pretrained": True, "time_init":  config['arch']['args']['video_params']['time_init']}
    text_params = {"model": config['arch']['args']['text_params']['model'], "pretrained": True, "input": config['arch']['args']['text_params']['input']}
    projection_dim=config['arch']['args']["projection_dim"]
    if args.model_name is not None:
        load_checkpoint=args.model_name
    else:
        load_checkpoint=""
    projection='minimal'
    load_temporal_fix='bilinear'
    task_names = 'EgoNCE_ITM_MLM'
    norm_layer = None
    embed_dim=768

    with open('./Data/qa/' + args.dataset_split_type + '/answer_set.txt', 'r') as ansf:
        answers = ansf.readlines()
        args.output_dim = len(answers) # # output_dim == len(answers)

    model = FrozenInTime(video_params, text_params, projection_dim=projection_dim, load_checkpoint=load_checkpoint,
                         projection=projection, load_temporal_fix=load_temporal_fix,
                         task_names = task_names, norm_layer = norm_layer, embed_dim=embed_dim, model_dim=768, output_dim=args.output_dim).cuda(gpu)


    global_step = 0
    start_epoch = 0
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
                         optimizer,
                         num_warmup_steps = int(args.nepoch * len(dataloader_train) * 0.1),
                         num_training_steps = int(args.nepoch * len(dataloader_train)))


    if args.resume_finetune_model_path is not None:
        ckpt = torch.load(args.resume_finetune_model_path, map_location='cpu')
        new_ckpt_model = state_dict_data_parallel_fix(ckpt['model_state_dict'], model.state_dict())
        model.load_state_dict(new_ckpt_model)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        global_step = ckpt['global_step']
        start_epoch = (global_step+1)//len(dataloader_train)

    if args.test_only_model_path is not None:
        ckpt = torch.load(args.test_only_model_path, map_location='cpu')
        new_ckpt_model = state_dict_data_parallel_fix(ckpt['model_state_dict'], model.state_dict())
        model.load_state_dict(new_ckpt_model)


    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)


    scaler = torch.cuda.amp.GradScaler()
    ddp_args={}
    ddp_args['static_graph'] = True
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True, **ddp_args)

    with open('./Data/qa/' + args.dataset_split_type + '/all_reasoning_types.txt', 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]

    train_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)
    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)

    if args.test_only_model_path is not None:
        test_acc_calculator.reset()
        test_loss, test_acc = validate(model, dataloader_test, 0, args, gpu, acc_calculator=test_acc_calculator)
        acc_dct = test_acc_calculator.get_acc()
        if args.rank == 0:
            print("Evaluation Accuracy: ", acc_dct)
        return 0

    
    if args.rank == 0:
        args.basedir = os.path.join(args.basedir, args.model_name.split('/')[-1].split('.')[0])
        args.basedir = os.path.join(args.basedir, args.dataset_split_type)
        log_dir = os.path.join(args.basedir, TIMESTAMP, 'logs')
        os.makedirs(log_dir)

        with open(os.path.join(log_dir, 'argument.txt'), 'w') as f:
            for key, value in vars(args).items():
                f.write('%s:%s\n'%(key, value))
                print(key, value)

        log_file = open(os.path.join(log_dir, 'log.txt'), 'w')
        stats_file = open(os.path.join(log_dir, 'stats.txt'), 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
        os.makedirs(os.path.join(args.basedir, TIMESTAMP, 'ckpts'), exist_ok=True)


    best_overall_accuracy = -1
    unique_dict = torch.load('./reasoning_unique_cat.pth')

    for epoch in range(start_epoch, args.nepoch):
        sampler_train.set_epoch(epoch)
        model.train()
        train_acc_calculator.reset()
        pbar = tqdm(total = len(dataloader_train))

        for idx, (video_frames, text_tokens, attention_masks, answers, reasoning_type_lst) in tqdm(enumerate(dataloader_train)):

            optimizer.zero_grad()

            B, num_frame_per_video, C, H, W = video_frames.shape
            video_frames, text_tokens, attention_masks, answers = video_frames.cuda(gpu, non_blocking=True), text_tokens.cuda(gpu, non_blocking=True), attention_masks.cuda(gpu, non_blocking=True), answers.cuda(gpu, non_blocking=True)

            data_batch = {"video": video_frames, "text": {"input_ids": text_tokens, "attention_mask": attention_masks}}
            
            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast():
                    
                    logits = model.forward(data_batch)
                    logits = args.allgather(logits, gpu, args)
                    answers = args.allgather(answers, gpu, args)
                    loss = criterion(logits, answers.long())

                    max_len = 7
                    # pad all tensors to have same length
                    reasoning_type_lst = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value= -100) for x in reasoning_type_lst]
                    reasoning_type_lst = torch.stack(reasoning_type_lst).cuda(gpu, non_blocking=True)
                    reasoning_type_lst = args.allgather(reasoning_type_lst, gpu, args)
                    

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            reasoning_lst = []

            for _i in range(reasoning_type_lst.shape[0]):
                _list = []
                for _j in range(reasoning_type_lst.shape[1]):
                    if reasoning_type_lst[_i][_j] > -100:
                        _list.append([*unique_dict.keys()][int(reasoning_type_lst[_i][_j])])
                reasoning_lst.append(_list)

            reasoning_type_lst = reasoning_lst
            del reasoning_lst

            pred = torch.argmax(logits, dim=1)
            train_acc = sum(pred == answers) / B*args.world_size

            train_acc_calculator.update(reasoning_type_lst, pred, answers)

            pbar.update(1)

            args.i_test = args.i_val = args.i_weight = len(dataloader_train)
            

            if args.rank == 0:
                if global_step % args.i_print  == 0:
                    print(f"global_step:{global_step}, train_loss:{loss.item()}, train_acc:{train_acc}")
                    stats = dict(epoch=epoch, step=global_step,
                             lr_weights=optimizer.param_groups[0]['lr'],
                             loss=loss.item(), train_acc=train_acc.item())
                    print(json.dumps(stats), file=stats_file)

            
            if (global_step+1) % args.i_val == 0:
                test_acc_calculator.reset()
                val_loss, val_acc = validate(model, dataloader_val, epoch, args, gpu, acc_calculator=test_acc_calculator)
                acc_dct = test_acc_calculator.get_acc()
                stats = dict(epoch=epoch, step=global_step, val_loss=val_loss.item(), val_acc=val_acc.item())
                for key, value in acc_dct.items():
                    stats.update({f'val/reasoning_{key}': value})
                
                if args.rank == 0:
                    print(json.dumps(stats), file=stats_file)
                    log_file.write(f'[VAL]: epoch: {epoch}, global_step: {global_step}\n')
                    log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')


            if (global_step+1) % args.i_test == 0:
                test_acc_calculator.reset()
                test_loss, test_acc = validate(model, dataloader_test, epoch, args, gpu, acc_calculator=test_acc_calculator)
                acc_dct = test_acc_calculator.get_acc()
                stats = dict(epoch=epoch, step=global_step, test_loss=test_loss.item(), test_acc=test_acc.item())
                
                for key, value in acc_dct.items():
                    stats.update({f'test/reasoning_{key}': value})
                    if key == args.dataset_split_type:
                        if best_overall_accuracy < value:
                            best_act_dict = copy.deepcopy(acc_dct)
                            best_overall_accuracy = value
                            if args.rank == 0:
                                torch.save({'model_state_dict': model.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'loss': loss,
                                            'global_step': global_step,}, os.path.join(args.basedir, TIMESTAMP, 'ckpts', f"model_best.tar"))

                
                if args.rank == 0:
                    print(json.dumps(stats), file=stats_file)
                    log_file.write(f'[TEST]: epoch: {epoch}, global_step: {global_step}\n')
                    log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')

            
            if (global_step+1) % args.i_weight == 0:
                if args.rank == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'global_step': global_step,
                    }, os.path.join(args.basedir, TIMESTAMP, 'ckpts', f"model_{global_step}.tar"))

            global_step += 1


        if args.rank == 0:
            acc_dct = train_acc_calculator.get_acc()
            for key, value in acc_dct.items():
                stats.update({f'train/reasoning_{key}': value})
            print(json.dumps(stats), file=stats_file)

        
            log_file.write(f'[TRAIN]: epoch: {epoch}, global_step: {global_step}\n')
            log_file.write(f'true count dct: {train_acc_calculator.true_count_dct}\nall count dct: {train_acc_calculator.all_count_dct}\n\n')
            log_file.write(f'============================ Printing Best Overall Accuracy till Now ==========================================\n\n')
            log_file.write(f'BEST RESULTS: {best_act_dict}\n\n')
            log_file.write(f'===============================================================================================================\n\n')
            log_file.flush()
        
def validate(model, val_dataloader, epoch, args, gpu, acc_calculator):
    model.eval()
    all_acc = 0
    all_loss = 0
    acc_calculator.reset()
    
    print('validating...')
    unique_dict = torch.load('./reasoning_unique_cat.pth')
    with torch.no_grad():
        starttime = time.time()
        for idx, (video_frames, text_tokens, attention_masks, answers, reasoning_type_lst) in tqdm(enumerate(val_dataloader)):

            B, num_frame_per_video, C, H, W = video_frames.shape
            video_frames, text_tokens, attention_masks, answers = video_frames.cuda(gpu, non_blocking=True), text_tokens.cuda(gpu, non_blocking=True), attention_masks.cuda(gpu, non_blocking=True), answers.cuda(gpu, non_blocking=True)

            data_batch = {"video": video_frames, "text": {"input_ids": text_tokens, "attention_mask": attention_masks}}
            logits = model(data_batch)

            logits_all = [torch.zeros_like(logits) for _ in range(args.world_size)]
            torch.distributed.all_gather(logits_all, logits)
            logits_all = torch.cat(logits_all, dim=0)

            answers_all = [torch.zeros_like(answers) for _ in range(args.world_size)]
            torch.distributed.all_gather(answers_all, answers)
            answers_all = torch.cat(answers_all, dim=0)

            all_loss += nn.CrossEntropyLoss().cuda(gpu)(logits_all, answers_all.long())
            # print('validate finish in', (time.time() - starttime) * (len(val_loader) - i), 's')
            # starttime = time.time()
            pred_all = torch.argmax(logits_all, dim=1)
            test_acc = sum(pred_all == answers_all) / B * args.world_size
            all_acc += test_acc


            max_len = 7
            # pad all tensors to have same length
            reasoning_type_lst = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value= -100) for x in reasoning_type_lst]
            reasoning_type_lst = torch.stack(reasoning_type_lst).cuda(gpu, non_blocking=True)
            reasoning_type_lst = args.allgather(reasoning_type_lst, gpu, args)

            reasoning_lst = []

            for _i in range(reasoning_type_lst.shape[0]):
                _list = []
                for _j in range(reasoning_type_lst.shape[1]):
                    if reasoning_type_lst[_i][_j] > -100:
                        _list.append([*unique_dict.keys()][int(reasoning_type_lst[_i][_j])])
                reasoning_lst.append(_list)

            acc_calculator.update(reasoning_lst, pred_all, answers_all)


    all_loss /= len(val_dataloader)
    all_acc /= len(val_dataloader)
    model.train()
    return all_loss, all_acc


def main():
    args = parse_args()
    args.allgather = AllGather_multi.apply
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


if __name__ == '__main__':
    main()
