# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist

from base import BaseTrainer, Multi_BaseTrainer_dist
from model.model import sim_matrix
from utils import inf_loop
import os
from csv import reader
from pathlib import Path
import sys
import time
import json 
import copy
import torch.nn.functional as F

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

class Multi_Trainer_dist_Charades(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, scheduler, gpu, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, scheduler, gpu, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.args.model_name = self.config["arch"]["args"]["load_checkpoint"].split("/")[-1]

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch, scaler, gpu):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """


        if dist.get_rank() == 0:
            Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)
            stats_file = open(Path(self.args.save_dir) / str(self.args.model_name + "_" + 'stats_vtc.txt'), 'a', buffering=1)
            print(' '.join(sys.argv))
            print(' '.join(sys.argv), file=stats_file)


        log = {}
        
        if self.do_validation and epoch == 1:
            val_log = self._valid_epoch(0, gpu)
            if self.args.rank == 0:
                log.update(val_log)

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding='max_length', max_length=15,
                                                  truncation=True) 
                data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                data['video'] = data['video'].cuda(gpu, non_blocking=True)

                self.optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():
                        loss, loss_dict, ret = self.model(data, self.allgather, self.n_gpu, self.args, self.config, self.loss, gpu, task_names='Dual', dataset_name='charades')

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()

                if self.args.rank == 0:
                    if batch_idx % self.args.print_freq == 0:
                        #print('Train Step: [{}/{}] Loss: {:.4f}'.format(batch_idx+1, len(train_loader), loss.item()))
                        stats = dict(epoch=epoch, step=batch_idx,
                                 lr_weights=self.optimizer.param_groups[0]['lr'],
                                 loss=loss.item())
                        print("stats: ", stats)
                        print(json.dumps(stats), file=stats_file)


                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch-1) * total + current
                    self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

                total_loss[dl_idx] += loss.detach().item()

                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.args.print_freq == 0 and self.args.rank == 0:
                    self.logger.info('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))
                
                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log.update({f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))})

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch, gpu)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch, gpu):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        target_arr = {x: [] for x in range(len(self.valid_data_loader))}

        # construct set of sentences.
        cls_arr = []
        with open('/cis/home/shraman/works_meta_2022/Datasets/CharadesEgo_v1_480/CharadesEgo/Charades_v1_classes.txt',
                  'r') as charades:
            csv_reader = list(reader(charades))
        for line in csv_reader:
            cls_arr.append(str(line[0][5:]))

        with torch.no_grad():
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            data_cls = self.tokenizer(cls_arr, return_tensors='pt', padding='max_length', max_length=30, truncation=True)
            data_cls = {key: val.cuda(gpu, non_blocking=True) for key, val in data_cls.items()}

            dict_cls = {'text': data_cls, 'video': torch.Tensor(1, self.config["data_loader"][0]["args"]["video_params"]["num_frames"], 3, 224, 224).cuda(gpu, non_blocking=True)} ## 'video' can be random here with proper shape
            ret = self.model.module.infer(dict_cls, return_embeds=True, task_names="Dual", ret={})
            text_embed = ret['text_embeds']
            text_embeds = text_embed.cpu().detach()

            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    meta_arr[dl_idx].append(data['meta'])
                    
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding='max_length', max_length=30, truncation=True)
                    data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                    data['video'] = data['video'].cuda(gpu, non_blocking=True)
                    data_target = data['target'].cuda(gpu, non_blocking=True)

                    ret = self.model.module.infer(data, return_embeds=True, task_names="Dual", ret={})
                    vid_embed = ret['video_embeds'].contiguous()

                    vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(vid_embed_all, vid_embed)
                    vid_embed_all = torch.cat(vid_embed_all, dim=0)
                    vid_embed_arr[dl_idx].append(vid_embed_all.cpu())

                    data_target_all = [torch.zeros_like(data_target) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_target_all, data_target)
                    data_target_all = torch.cat(data_target_all, dim=0)
                    target_arr[dl_idx].append(data_target_all.cpu())

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch-1)

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            target_embeds = torch.cat(target_arr[dl_idx])

            sims = sim_matrix(text_embeds, vid_embeds).numpy().T
            pred_arr_cat = {"charades_metrics_vtc": sims}

            targets = target_embeds.numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(pred_arr_cat[metric_name], targets)
                if self.args.rank == 0:
                    self.logger.info( verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name, args=self.args) )
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        key = key.replace('[', '_').replace(']', '_')
                        self.writer.add_scalar(f'Val_metrics_{dl_idx}/{key}', val, epoch-1)

                if self.visualizer is not None and self.args.rank == 0:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                        for dl_idx in range(len(self.valid_data_loader))}
            res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

def verbose(epoch, metrics, mode, args, name="TEST"):
    
    if dist.get_rank() == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        stats_file = open(Path(args.save_dir) / str(args.model_name + "_" + 'stats_vtc.txt'), 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    
    mAP = metrics["mAP"]
    msg = f"[{mode}]{name:s} epoch {epoch}, mAP: {mAP:.3f}"
    print(msg)

    if dist.get_rank()==0:
        #print('Train Step: [{}/{}] Loss: {:.4f} Time: {}'.format(step+1, len(train_loader), loss.item(), int(time.time() - start_time)))
        stats = dict(epoch=epoch, msg=msg)
        print(json.dumps(stats), file=stats_file)
    
    return msg

def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
