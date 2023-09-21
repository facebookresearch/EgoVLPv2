# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import pdb

from utils.data_util import pad_seq, pad_char_seq, pad_video_seq
from utils.data_gen import visual_feature_sampling

class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_root_list, args):
        super(Dataset, self).__init__()
        self.feature_root_list = feature_root_list 
        self.args = args

    def __getitem__(self, index):

        feature = torch.load(self.feature_root_list[index])
        video_feature = feature["fused_video_features"]
        video_feature = visual_feature_sampling(video_feature, max_num_clips=self.args.max_pos_len)
        dual_text_feature = feature["dual_text_features"][0]
        query_attention_mask = feature["text_token"]["attention_mask"][0]
        s_ind, e_ind = int(feature["s_ind"].item()), int(feature["e_ind"].item())
        return feature, video_feature, dual_text_feature, s_ind, e_ind, query_attention_mask

    def __len__(self):
        return len(self.feature_root_list)


def train_collate_fn(data):

    feature, video_features, dual_text_feature, s_inds, e_inds, query_attention_mask = zip(*data)

    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)

    #for _idx in range(len(vfeats)):
    #    print(vfeats[_idx].shape)

    #pdb.set_trace()

    vfeats = np.asarray([vf.numpy() for vf in vfeats], dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0.1
    for idx in range(batch_size):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_ : (et_ + 1)] = 1
        else:
            h_labels[idx][st : (et + 1)] = 1
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    dual_text_feature = torch.stack(list(dual_text_feature), dim=0)
    query_attention_mask = torch.stack(list(query_attention_mask), dim=0)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    h_labels = torch.tensor(h_labels, dtype=torch.int64)
    return feature, vfeats, vfeat_lens, dual_text_feature, s_labels, e_labels, h_labels, query_attention_mask


def test_collate_fn(data):

    feature, video_features, dual_text_feature, _, _, query_attention_mask = zip(*data)

    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray([vf.numpy() for vf in vfeats], dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # convert to torch tensoet number

    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    dual_text_feature = torch.stack(list(dual_text_feature), dim=0)
    query_attention_mask = torch.stack(list(query_attention_mask), dim=0)
    return feature, vfeats, vfeat_lens, dual_text_feature, query_attention_mask


def get_train_loader(feature_root_list, args):
    train_set = Dataset(feature_root_list=feature_root_list, args=args)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
    )
    return train_loader


def get_test_loader(feature_root_list, args):
    test_set = Dataset(feature_root_list=feature_root_list, args=args)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
    )
    return test_loader
