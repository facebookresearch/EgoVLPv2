# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re
import pdb


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 200):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len



class Transformer_Encoder(nn.Module):
    def __init__(self, model_dim, nhead, num_layers):
        super(Transformer_Encoder, self).__init__()

        self.model_dim = model_dim
        self.pos = PositionalEmbedding(model_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.projector_1 = nn.Linear(model_dim, 8)
        self.summ_head = nn.Linear(8, 1)

    def create_mask(self, seg_len):
        
        masked_seg = torch.empty(seg_len.shape[0], seg_len.shape[1], 200)
        for batch in range(seg_len.shape[0]):
            for idx in range(len(seg_len[batch])):
                zeros = torch.zeros(seg_len[batch][idx].item())
                ones = torch.ones(200-seg_len[batch][idx].item())
                mask = torch.cat((zeros, ones))
                masked_seg[batch][idx] = mask
        return masked_seg #(batch_size, 20, 200)


    def forward(self, features, seg_len):

        # features -> (batch_size, 20, 200, 768)

        batch_size = features.shape[0]
        features = features.reshape(features.shape[0]*features.shape[1], features.shape[2], features.shape[3])

        features = features + self.pos(features) #(batch_size*20, 200, model_dim)

        masked_seg = self.create_mask(seg_len).reshape(self.create_mask(seg_len).shape[0]*self.create_mask(seg_len).shape[1], self.create_mask(seg_len).shape[2]).to(features.device)  #(batch_size*20, 200)

        features = features.permute(1,0,2)
        batch_out = self.transformer_encoder(features, src_key_padding_mask = masked_seg)
        batch_out = batch_out.permute(1,0,2) #(batch_size*20, 200, model_dim)

        batch_out = batch_out.reshape(batch_size, features.size(1), batch_out.shape[1], batch_out.shape[2])
        batch_out = self.summ_head(self.dropout(self.relu(self.projector_1(batch_out)))).squeeze(-1)

        return batch_out
