# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import argparse
import collections
import transformers
import signal
import subprocess
from set_optim_schedule import set_schedule
import yaml

with open('./EgoNCE_MLM_ITM_Config.yml') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)

import torch
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model_epic_charades as module_arch
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer.trainer_epic import Multi_Trainer_dist_MIR
from utils.util import replace_nested_dict_item
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('--task_names', default='EgoNCE_ITM_MLM', type=str, help='Task_Names')
parser.add_argument('-c', '--config', default='/fsx/spraman3/Video_Language_Pretraining/Pre-training/EgoVLP_multinode/configs/pt/egoclip.json', type=str,
                help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                help='indices of GPUs to enable (default: all)')
parser.add_argument('-o', '--observe', action='store_true',
                help='Whether to observe (neptune)')
parser.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
parser.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
parser.add_argument('-sc', '--schedule', default=[60, 80])
parser.add_argument('--print_freq', type=int, default=100, help="print loss after this number of steps")
parser.add_argument('--save_dir', type=str, help="dirctory for model saving")

CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
config = ConfigParser(parser)


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

def main():
    
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        print("If is being executed")
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        print("Else is being executed")
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):

    print("main worker started")
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    print("init processed finished")

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    print("device set")

    
    logger = config.get_logger('train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None
        
    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    
    if args.rank == 0:
        print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
        print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')
    # build model architecture, then print to console
    
    print("dataloader initialized")

    
    model = config.initialize('arch', module_arch)

    if args.rank == 0:
        logger.info(model)

    
    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    
    ## set_schedule will be modified; currently same as pretraining
    max_steps = int(len(data_loader[0]) * config['trainer']['epochs'])
    if max_steps==0:
        max_steps = int(len(data_loader[0]) * 10)
    warmup_steps = config_yaml["warmup_steps"]
    if isinstance(config_yaml["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    optimizer, scheduler = set_schedule(model, config, config_yaml, max_steps, warmup_steps)
    
    lr_scheduler = None
    writer = None

    if args.rank == 0:
        writer = SummaryWriter(log_dir=str(config.tf_dir))

    print("trainer should being here")
    trainer = Multi_Trainer_dist_MIR(args, model, loss, metrics, optimizer, scheduler, gpu,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])

    
    trainer.train(gpu)
    

def init_dataloaders(config, module_data):
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', 'val')
        valid_data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                       range(len(config['data_loader']))]
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'val')
            new_cfg_li.append(dl_cfg)
        config._config['data_loader'] = new_cfg_li
        valid_data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                             range(len(config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


if __name__ == '__main__':
    main()
