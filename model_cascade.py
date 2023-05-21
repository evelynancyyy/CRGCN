#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: yanms
# @Date  : 2021/11/1 16:16
# @Desc  : CRGCN
import json
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
# from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss


class SpAdjEdgeDrop(nn.Module):
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edge_num = vals.size()
        mask = (torch.rand(edge_num) + keep_rate).floor().type(torch.bool)
        new_vals = vals[mask]
        new_idxs = idxs[:, mask]
        return torch.sparse.FloatTensor(new_idxs, new_vals, adj.shape)


class LightGCN(nn.Module):
    def __init__(self, layers, dropout):
        super(LightGCN, self).__init__()
        self.layers = layers
        self.edge_dropper = SpAdjEdgeDrop()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj, keep_rate):
        adj = self.edge_dropper(adj, keep_rate)
        all_embeddings = []
        for i in range(self.layers):
            x = torch.sparse.mm(adj, x)
            all_embeddings.append(x)
            # x = self.dropout(x)
        x = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        return x


class CRGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(CRGCN, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        self.keep_rate = args.keep_rate
        self.behavior_adjs = dataset.behavior_adjs
        self.behavior_cf = nn.ModuleDict({
            behavior: LightGCN(self.layers[index], self.node_dropout) for index, behavior in enumerate(self.behaviors)
        })

        self.reg_weight = args.reg_weight
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.if_multi_tasks = args.if_multi_tasks

        self.storage_all_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

        # TODO
        with open(os.path.join(args.data_path, 'global_ratio.txt'), 'r', encoding='utf-8') as f:
            global_ratio = json.load(f)
        self.global_ratio = global_ratio

        with open(os.path.join(args.data_path, 'user_behavior_ratio.txt'), 'r', encoding='utf-8') as f:
            user_behavior_ratio = json.load(f)
        self.user_behavior_ratio = user_behavior_ratio  # user: [r1,r2], r3=1不存了

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self):
        """
        gcn propagate in each behavior
        """
        all_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for behavior in self.behaviors:
            layer_embeddings = total_embeddings
            layer_adj = self.behavior_adjs[behavior].to(self.device)
            layer_embeddings = self.behavior_cf[behavior](layer_embeddings, layer_adj, self.keep_rate)  # lightGCN 4 b
            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            total_embeddings = layer_embeddings + total_embeddings
            all_embeddings[behavior] = total_embeddings
        return all_embeddings

    # 源码
    # def forward(self, batch_data):
    #     self.storage_all_embeddings = None
    #
    #     all_embeddings = self.gcn_propagate()  # dict (|B|, N, dim)
    #     total_loss = 0
    #     for index, behavior in enumerate(self.behaviors):
    #         if self.if_multi_tasks or behavior == 'buy':
    #             data = batch_data[:, index]  # (bsz,3)
    #             users = data[:, 0].long()  # (bsz,)
    #             items = data[:, 1:].long()  # (bsz, 2)
    #             user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])
    #
    #             user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)  # (bsz, 2, dim )
    #             item_feature = item_all_embedding[items]  # (bsz, 2, dim)
    #             # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)
    #
    #             scores = torch.sum(user_feature * item_feature, dim=2)  # (bsz, 2)
    #             total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])
    #     total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)
    #
    #     return total_loss

    def forward(self, batch_data):
        self.storage_all_embeddings = None

        all_embeddings = self.gcn_propagate()  # dict (|B|, N, dim)
        total_loss = 0
        for index, behavior in enumerate(self.behaviors):
            if self.if_multi_tasks or behavior == 'buy':
                data = batch_data[:, index]  # (bsz,3)
                users = data[:, 0].long()  # (bsz,)
                items = data[:, 1:].long()  # (bsz, 2)
                user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior],
                                                                     [self.n_users + 1, self.n_items + 1])

                user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)  # (bsz, 2, dim )
                item_feature = item_all_embedding[items]  # (bsz, 2, dim)
                # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)

                user_ratio = self.get_user_ratio(behavior, users)
                user_ratio = user_ratio.to(self.device)

                scores = torch.sum(user_feature * item_feature, dim=2)  # (bsz, 2)
                temp = (user_ratio * self.bpr_loss(scores[:, 0], scores[:, 1])).mean()  # 哈达马积，然后求平均
                total_loss += temp
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight,
                                                                  self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]],
                                                     [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]  # (test_bsz, dim)
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))  # (test_bsz, |I|)
        return scores

    def get_user_ratio(self, behavior, users):
        user_behavior_ratio = self.user_behavior_ratio
        total_ratio_list = []
        # 找user对应的ratio_list，组合成array (len(users),3)
        for u in range(len(users)):
            ratio_list = user_behavior_ratio[str(int(users[u].item()))]
            ratio_list.append(1)
            total_ratio_list.append(ratio_list)

        ratio_array = np.array(total_ratio_list)
        ratio_array = ratio_array.T  # (3,len(users))

        if behavior == 'click':
            ratio = ratio_array[0]  # array
            ratio = torch.from_numpy(ratio)  # tensor

        if behavior == 'cart':
            ratio = ratio_array[1]  # array
            ratio = torch.from_numpy(ratio)  # tensor

        if behavior == 'buy':
            ratio = ratio_array[2]  # array
            ratio = torch.from_numpy(ratio)  # tensor

        assert len(ratio) == len(users)
        return ratio
