# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.distributed as dist
import copy
import torch.nn.functional as F
from pathlib import Path
import sys
import json 

from base import Multi_BaseTrainer_dist
from model.model import sim_matrix, sim_matrix_batch_val
from utils import inf_loop
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling


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

class Multi_Trainer_dist(Multi_BaseTrainer_dist):
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
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = self.args.print_freq #int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=0.15)
        # self.writer = writer

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            # if self.writer is not None:
            #     self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics


    def _train_epoch(self, epoch, scaler, gpu):

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))

        if dist.get_rank() == 0:
            Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)
            stats_file = open(Path(self.args.save_dir) / 'stats.txt', 'a', buffering=1)
            print(' '.join(sys.argv))
            print(' '.join(sys.argv), file=stats_file)


        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            #print("Length of data_li: ", len(data_li))
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                
                if 'video_neg' in data.keys():  # w/ negative sampling
                    data['text'] = data['text'] + data['text_neg']
                    data['video'] = torch.cat( (data['video'], data['video_neg']), axis = 0)
                    data['noun_vec'] = torch.cat((data['noun_vec'], data['noun_vec_neg']), axis=0)
                    data['verb_vec'] = torch.cat((data['verb_vec'], data['verb_vec_neg']), axis=0)


                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding='max_length', max_length=15,
                                                  truncation=True)

                if 'MLM' in self.args.task_names:
                    
                    data_text_input_list = []
                    for idx in range(data['text']['input_ids'].size(0)):
                        data_text_input_list.append(data['text']['input_ids'][idx])
                    mlm_dict = self.data_collator(data_text_input_list)


                    data.update({'text_mlm_ids': mlm_dict['input_ids'], 'text_mlm_labels': mlm_dict['labels']})
                    data['text_mlm_ids'] = data['text_mlm_ids'].cuda(gpu, non_blocking=True)
                    data['text_mlm_labels'] = data['text_mlm_labels'].cuda(gpu, non_blocking=True)

                data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                data['video'] = data['video'].cuda(gpu, non_blocking=True)
                n_embeds = data['noun_vec'].cuda(gpu, non_blocking=True)
                v_embeds = data['verb_vec'].cuda(gpu, non_blocking=True)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():

                        loss, loss_dict, ret = self.model(data, n_embeds, v_embeds, self.allgather, self.n_gpu, self.args, self.config, self.loss, gpu, task_names=self.args.task_names)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()


                if dist.get_rank()==0:
                    if (batch_idx) % self.args.print_freq == 0:
                        #print('Train Step: [{}/{}] Loss: {:.4f} Time: {}'.format(step+1, len(train_loader), loss.item(), int(time.time() - start_time)))
                        stats = dict(epoch=epoch, step=batch_idx,
                                    lr_weights=self.optimizer.param_groups[0]['lr'],
                                    loss=loss.item())
                        print(json.dumps(stats), file=stats_file)



                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch-1) * total + current
                    self.writer.add_scalar(f'Loss_training/loss_{dl_idx}', loss.detach().item(), final_total)

                total_loss[dl_idx] += loss.detach().item()

                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.log_step == 0 and self.args.rank == 0:
                    self.logger.info('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))

                self.optimizer.zero_grad()
            
            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(f'Loss_training/loss_total_{dl_idx}', tl, epoch-1)

        if self.do_validation:
            val_log = self._valid_epoch(epoch, gpu)
            if self.args.rank == 0:
                log.update(val_log)


        return log

    def _valid_epoch(self, epoch, gpu):
        
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)

        gt_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pred_arr_ensemble = {x: [] for x in range(len(self.valid_data_loader))}
        pred_arr_vtm = {x: [] for x in range(len(self.valid_data_loader))}
        type_arr = {x: [] for x in range(len(self.valid_data_loader))}

        

        with torch.no_grad():
            for dl_idx, dl in enumerate(self.valid_data_loader):

                for batch_idx, data in enumerate(tqdm(dl)):
                    b1,b2,b3,b4,b5,b6= data['video'].shape #b1 -> batch_size, b2 -> 5, b3 -> 4
                    data['video'] = data['video'].reshape(b1*b2,b3,b4,b5,b6)
                    data['text'] = data['text']

                    
                    data_text_all = []
                    for _i in range(self.n_gpu):
                        if self.args.rank == _i:
                            data_text_all.append(data['text'])


                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.cuda(gpu, non_blocking=True) for key, val in data['text'].items()}
                    data['video'] = data['video'].cuda(gpu, non_blocking=True)


                    ret = self.model.module.infer(data, return_embeds=True, task_names="EgoNCE", ret={})

                    data['text']['input_ids'] = torch.repeat_interleave(data['text']['input_ids'], b2, dim=0)
                    data['text']['attention_mask'] = torch.repeat_interleave(data['text']['attention_mask'], b2, dim=0)

                    ret = self.model.module.infer(data, return_embeds=True, task_names="ITM", ret=ret)

                    data_gt = data['correct'][0].cuda(gpu, non_blocking=True).unsqueeze(0)


                    ret['text_embeds'] = ret['text_embeds'].reshape(b1,1,ret['text_embeds'].shape[-1])
                    ret['video_embeds'] = ret['video_embeds'].reshape(b1,b2,ret['video_embeds'].shape[-1])
                    data_pred_vtc = sim_matrix_batch_val(ret['text_embeds'], ret['video_embeds']).squeeze(1)
                    data_pred_vtm = F.softmax(ret["cross_attn_itm_logits"],dim=1)[:,1:].t().reshape(1,b1,b2)[0].contiguous()
                    
                    data_type = data['type'][0].cuda(gpu, non_blocking=True).unsqueeze(0)
                    data_gt_all = [torch.zeros_like(data_gt) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_gt_all, data_gt)
                    data_gt_all = torch.cat(data_gt_all, dim=0)

                    data_pred_all_vtc = [torch.zeros_like(data_pred_vtc) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_pred_all_vtc, data_pred_vtc)
                    data_pred_all_vtc = torch.cat(data_pred_all_vtc, dim=0)

                    data_pred_all_vtm = [torch.zeros_like(data_pred_vtm) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_pred_all_vtm, data_pred_vtm)
                    data_pred_all_vtm = torch.cat(data_pred_all_vtm, dim=0)

                    #print("Shape of data_pred_all_vtc: ", data_pred_all_vtc.shape)
                    #print("Shape of data_pred_all_vtm: ", data_pred_all_vtm.shape)
                    data_pred_all_ensemble = data_pred_all_vtc + data_pred_all_vtm

                    data_type_all = [torch.zeros_like(data_type) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_type_all, data_type)
                    data_type_all = torch.cat(data_type_all, dim=0)

                    gt_arr[dl_idx].append(data_gt_all.cpu())
                    pred_arr_ensemble[dl_idx].append(data_pred_all_ensemble.cpu())
                    pred_arr_vtm[dl_idx].append(data_pred_all_vtm.cpu())
                    type_arr[dl_idx].append(data_type_all.cpu())


            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(f'Loss_val/loss_total_{dl_idx}', tl, epoch-1)

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            gt_arr_cat = torch.cat(gt_arr[dl_idx])
            pred_arr_cat = {"egomcq_accuracy_metrics_ensemble": torch.cat(pred_arr_ensemble[dl_idx]), "egomcq_accuracy_metrics_vtm": torch.cat(pred_arr_vtm[dl_idx])}
            type_cat = torch.cat(type_arr[dl_idx])

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(pred_arr_cat[metric_name], gt_arr_cat, type_cat)
                if self.args.rank == 0:
                    self.logger.info(
                        verbose(epoch=epoch, metrics=res, args=self.args, name=self.valid_data_loader[dl_idx].dataset_name))
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    # for key, val in to_write.items():
                    #     self.writer.log_scalar(key, val)
                    for key, val in to_write.items():
                        key = key.replace('[', '_').replace(']', '_')
                        self.writer.add_scalar(f'Val_metrics_{dl_idx}/{key}', val, epoch - 1)

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

def verbose(epoch, metrics, args, name="TEST"):
    msg = ""
    if dist.get_rank() == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        stats_file = open(Path(args.save_dir) / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    
    for key in metrics.keys():
        acc = metrics[key]
        msg += f"{name:s} epoch {epoch}, {key:s}, Acc: {acc:.1f};    "
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
