# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch

path = '/apdcephfs/private_qinghonglin/video_codebase/frozen-in-time-main/results/EgoClip_M_EgoNCE_N_V_Neg_Seg_60/models/0509_00/checkpoint-epoch3.pth'
# path = '/apdcephfs/private_qinghonglin/video_codebase/frozen-in-time-main/results/EgoClip_SE_1_mid_scale_th_v2/models/0502_01/checkpoint-epoch4.pth'
checkpoint = torch.load(path)
state_dict = checkpoint['state_dict']

