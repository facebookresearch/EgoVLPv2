# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb

import timm
import torch
import yaml
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange, repeat

from base import BaseModel
from model import video_transformer
from model.video_transformer import SpaceTimeTransformer
from utils.util import state_dict_data_parallel_fix

from model import roberta
from model.roberta import RobertaModel, _prepare_decoder_attention_mask
from model import heads
from transformers import RobertaConfig
from functools import partial
import copy
import torch.distributed as dist
import pickle

with open('./EgoNCE_MLM_ITM_Config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class FrozenInTime(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=4096,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='bilinear',
                 config = config,
                 task_names = 'EgoNCE_ITM_MLM',
                 norm_layer = None,
                 embed_dim=768):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        self.config = config
        self.task_names = task_names
        try:
            self.relevancy_mat_train_path = os.path.join(self.config['data_loader']['args']['meta_dir'], 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl')
            pkl_file = open(self.relevancy_mat_train_path, 'rb')
            self.relevancy_mat_train = pickle.load(pkl_file)
        except:
            pass
        
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        if self.text_params['model'].startswith('roberta'):
            self.text_model = RobertaModel.from_pretrained("roberta-base")
        self.text_model.train()

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            self.num_frames = video_params["num_frames"]
            self.drop_path_rate = video_params["drop_path_rate"]
            time_init = 'zeros'
            attention_style = 'frozen-in-time'
            arch_config = 'base_patch16_224'
            vit_init = 'imagenet-21k'
            if arch_config == 'base_patch16_224':
                #vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                vit_model = torch.load("/cis/home/shraman/works_meta_2022/pre-training/EgoVLP_Fused_HardNegITM_Checkpoint_multinode/frozen-in-time-main/pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                model = SpaceTimeTransformer(num_frames=self.num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style,
                                            drop_path_rate=self.drop_path_rate)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
           
            if load_checkpoint in ["", None]:
                # vit_checkpoint = vit_model.state_dict()
                # model.load_state_dict(vit_checkpoint, strict=False)
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
                model.load_state_dict(new_vit_dict, strict=False)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':

            txt_proj = nn.Sequential(nn.ReLU(), nn.Linear(self.text_model.config.hidden_size, 256),)
            vid_proj = nn.Sequential(nn.Linear(ftr_dim, 256))

            #txt_proj = nn.Sequential(
            #    nn.Linear(self.text_model.config.hidden_size, projection_dim, bias=False),
            #    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
            #    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
            #)

            #vid_proj = nn.Sequential(
            #    nn.Linear(ftr_dim, projection_dim, bias=False),
            #    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
            #    nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
            #)

        
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if ('MLM' in self.task_names or 'ITM' in self.task_names):
            # for FIBER-like cross-attention

            bert_config = RobertaConfig(
                vocab_size=self.config["vocab_size"],
                hidden_size=self.config["hidden_size"],
                num_hidden_layers=self.config["num_layers"],
                num_attention_heads=self.config["num_heads"],
                intermediate_size=self.config["hidden_size"] * config["mlp_ratio"],
                #max_position_embeddings=maxlen, [was used in BTGOT script]
                hidden_dropout_prob=self.config["drop_rate"],
                attention_probs_dropout_prob=self.config["drop_rate"],
            )

            self.num_fuse_block=self.config["num_fuse_block"]
            self.num_text_layer=self.config["num_layers"]
            roberta.NUM_FUSE_BLOCK = self.video_model.NUM_FUSE_BLOCK=self.num_fuse_block
            roberta.DIM_IMG=self.config["input_image_embed_size"]
            self.video_model.DIM_TXT=self.config["input_text_embed_size"]

            self.cross_modal_text_transform = nn.Linear(self.config["input_text_embed_size"], self.config["hidden_size"])
            self.cross_modal_text_transform.apply(init_weights)
            self.cross_modal_video_transform = nn.Linear(self.config["input_image_embed_size"], self.config["hidden_size"])
            self.cross_modal_video_transform.apply(init_weights)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.num_patches = self.video_model.patch_embed.num_patches
            self.patches_per_frame = self.num_patches//self.num_frames
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)
            self.pre_logits = nn.Identity()


            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.cross_modal_video_pooler = heads.Pooler(config["hidden_size"])
            self.cross_modal_video_pooler.apply(init_weights)
            self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
            self.cross_modal_text_pooler.apply(init_weights)

            ## einops transformations
            self.einops_from_space = 'b (f n) d'
            self.einops_to_space = '(b f) n d'
            self.einops_from_time = 'b (f n) d'
            self.einops_to_time = '(b n) f d'

        if 'MLM' in self.task_names:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(init_weights)

        if 'ITM' in self.task_names:
            self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            self.itm_score.apply(init_weights)

        if load_checkpoint not in ["", None]:
            # checkpoint = torch.load(load_checkpoint)
            # local_rank = int(os.environ['LOCAL_RANK'])  # fixed by qinghong.
            # checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(local_rank))
            checkpoint = torch.load(load_checkpoint, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=False)

    def set_device(self, device):
        self.device = device

    def infer(self, data, video_only=False, return_embeds=True, task_names=None, ret={}):

        text_data = data['text']
        video_data = data['video']

        if task_names is not None:
            self.task_names = task_names


        if 'Dual' in self.task_names: 

            text_embeddings = self.compute_text(text_data)
            video_embeddings = self.compute_video(video_data)

            if return_embeds:
                ret.update({'text_embeds':text_embeddings,
                'video_embeds':video_embeddings
                })

        if 'ITM' in self.task_names:

            b, curr_frames, channels, _, _ = video_data.shape
            video_data_itm = self.video_model.patch_embed(video_data)
            video_data_itm = video_data_itm.flatten(2).transpose(2, 1)
            video_data_itm = video_data_itm.reshape(b, -1, self.video_model.patch_embed.embed_dim)

            BF = video_data_itm.shape[0]
            cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            video_data_itm = torch.cat((cls_tokens, video_data_itm), dim=1)
            # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
            cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
            tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
            # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

            n = self.patches_per_frame
            f = curr_frames

            curr_patches = video_data_itm.shape[1]
            video_data_itm = video_data_itm + total_pos_embed[:, :curr_patches]
            video_data_itm = self.video_model.pos_drop(video_data_itm)

            unfused_blocks = self.num_text_layer - self.num_fuse_block

            

            for blk_i, blk in enumerate(self.video_model.blocks[:unfused_blocks]):
                if config['use_checkpoint']:
                    video_data_itm = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                n, f)
                else:
                    video_data_itm = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n, space_f=f)
                            
            
            text_embeds = self.text_model.embeddings(input_ids=text_data['input_ids']) # before it was input_ids=text_ids
            device = text_embeds.device
            #print(text_data.keys())
            text_masks = text_data['attention_mask']
            input_shape = text_masks.size()
            extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)
            for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
                if config['use_checkpoint']:
                    text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks)[0]
                else:
                    text_embeds = layer(text_embeds, extend_text_masks)[0]


            for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
                if config['use_checkpoint']:
                    

                    fuse_video_data = torch.utils.checkpoint.checkpoint(blk, video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
                                          n, f, text_embeds, extend_text_masks)
                    text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
                                          text_embeds, extend_text_masks, None, (video_data_itm), None, None, False, True)[0]
                else:
                    fuse_video_data = blk(video_data_itm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time, 
                                          y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f)
                    text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_itm), last_norm=True)[0]
                video_data_itm = fuse_video_data

            
            #print("Shape of model output", video_data.shape)
            video_data_itm = self.norm(video_data_itm)[:, 0]
            video_data_itm = self.pre_logits(video_data_itm)

            text_embeds = text_embeds[:, 0]
            text_embeds = self.cross_modal_text_transform(text_embeds)
            video_embeds = self.cross_modal_video_transform(video_data_itm)

            cls_feats_text = self.cross_modal_text_pooler(text_embeds)
            
            cls_feats_video = self.cross_modal_video_pooler(video_embeds)

            cls_feats = torch.cat([cls_feats_text, cls_feats_video], dim=-1)

            ret.update({
                "cross_attn_itm_logits": self.itm_score(cls_feats)
            })

            #print("ITM Forward Pass Done!!")


        if 'MLM' in self.task_names:
            
            #print("MLM started")

            b, curr_frames, channels, _, _ = video_data.shape
            video_data_mlm = self.video_model.patch_embed(video_data)
            video_data_mlm = video_data_mlm.flatten(2).transpose(2, 1)
            video_data_mlm = video_data_mlm.reshape(b, -1, self.video_model.patch_embed.embed_dim)

            #print("video_data_mlm shape: ", video_data_mlm.shape)

            BF = video_data_mlm.shape[0]
            cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            video_data_mlm = torch.cat((cls_tokens, video_data_mlm), dim=1)
            # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
            cls_embed = self.video_model.pos_embed[:, 0, :].unsqueeze(1)
            tile_pos_embed = self.video_model.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
            # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.video_model.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

            #print("total_pos_embed shape: ", total_pos_embed.shape)

            n = self.patches_per_frame
            f = curr_frames

            curr_patches = video_data_mlm.shape[1]
            video_data_mlm = video_data_mlm + total_pos_embed[:, :curr_patches]
            video_data_mlm = self.video_model.pos_drop(video_data_mlm)

            #print("video_data_mlm shape: ", video_data_mlm.shape)

            unfused_blocks = self.num_text_layer - self.num_fuse_block

            
            for blk_i, blk in enumerate(self.video_model.blocks[:unfused_blocks]):
                if config['use_checkpoint']:
                    video_data_mlm = torch.utils.checkpoint.checkpoint(blk, video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                n, f)
                else:
                    video_data_mlm = blk(video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                time_n=n, space_f=f)
            #print("video_data_mlm shape: ", video_data_mlm.shape)
                       
            
            text_embeds = self.text_model.embeddings(input_ids=data['text_mlm_ids']) # before it was input_ids=text_ids
            device = text_embeds.device
            text_masks = text_data['attention_mask']
            input_shape = text_masks.size()
            extend_text_masks = self.text_model.get_extended_attention_mask(text_masks, input_shape, device)

            for layer_i, layer in enumerate(self.text_model.encoder.layer[:unfused_blocks]):
                if config['use_checkpoint']:
                    text_embeds = torch.utils.checkpoint.checkpoint(layer, text_embeds, extend_text_masks)[0]
                else:
                    text_embeds = layer(text_embeds, extend_text_masks)[0]

            for blk_i, blk in enumerate(self.video_model.blocks[unfused_blocks:self.num_text_layer]):
                if config['use_checkpoint']:

                    fuse_video_data = torch.utils.checkpoint.checkpoint(blk, video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                            n, f, text_embeds, extend_text_masks)
                    text_embeds = torch.utils.checkpoint.checkpoint(self.text_model.encoder.layer[blk_i + unfused_blocks],
                                          text_embeds, extend_text_masks, None, (video_data_mlm), None, None, False, True)[0]
                else:
                    fuse_video_data = blk(video_data_mlm, self.einops_from_space, self.einops_to_space, self.einops_from_time, self.einops_to_time,
                                          y=text_embeds, y_mask=extend_text_masks, time_n=n, space_f=f)
                    text_embeds = self.text_model.encoder.layer[blk_i + unfused_blocks](text_embeds, extend_text_masks, encoder_hidden_states=(video_data_mlm), last_norm=True)[0]
                video_data_mlm = fuse_video_data


            text_embeds = text_embeds #[:, 0]
            text_embeds = self.cross_modal_text_transform(text_embeds)

            ret.update({
                "cross_attn_mlm_logits": self.mlm_score(text_embeds)
            })

        return ret
        #return sim_matrix(text_embeddings, video_embeddings)

    
    def create_mask_relevancy(self, relevancy_mat, item_v, item_t):

        mask_rel = torch.zeros(len(item_v), len(item_t))
        item_t_list = item_t.tolist()
        for _i in range(len(item_v)):
            current_item_v = int(item_v[_i].item())
            mask_rel[current_item_v] = relevancy_mat[current_item_v][item_t_list] > 0.1

        return mask_rel



    def forward(self, data, allgather, n_gpu, args, config, loss_dual, gpu, return_embeds=True, task_names='Dual', dataset_name='charades'):

        ret = {}
        loss_dict = {}

        if 'Dual' in task_names:
            
            ret = self.infer(data, task_names='Dual')
            video_embeds = ret['video_embeds']
            text_embeds = ret['text_embeds']
            video_embeds = allgather(video_embeds, n_gpu, args)
            text_embeds = allgather(text_embeds, n_gpu, args)

            output = sim_matrix(text_embeds, video_embeds)

            if dataset_name == 'epic':
                w_embeds = data['relation']
                w_embeds = allgather(w_embeds, n_gpu, args)
                loss = loss_dual(output, w_embeds)
            elif dataset_name == 'charades':
                loss, temp = loss_dual(output)
            else:
                raise NameError()

            if dataset_name == 'epic':
                ret.update({"sim_v2t": output, "sim_t2v": output.t(), 'epic_relation': w_embeds})
            elif dataset_name == 'charades':
                ret.update({"sim_v2t": output, "sim_t2v": output.t()})
            else:
                raise NameError()

            loss_dict.update({'Dual': loss})

        return loss, loss_dict, ret


    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        if config['use_checkpoint']:
            text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']    # not implement for bert
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        if config['use_checkpoint']:
            text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        if config['use_checkpoint']:
            video_embeddings = torch.utils.checkpoint.checkpoint(self.vid_proj, video_embeddings)
        else:
            video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sim_matrix_batch_val(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1).unsqueeze(-1), b.norm(dim=-1).unsqueeze(-1)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


if __name__ == "__main__":
    pass
