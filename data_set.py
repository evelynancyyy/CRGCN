#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author: yanms
# @Date  : 2021/11/1 11:38
# @Desc  :
import argparse
import os
import random
import json
import torch
import scipy.sparse as sp

from torch.utils.data import Dataset, DataLoader
import numpy as np

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []
        for behavior in self.behaviors:

            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                signal = [0, 0, 0]
            else:
                pos = random.sample(items, 1)[0]
                neg = random.randint(1, self.item_count)
                while np.isin(neg, self.behavior_dict['all'][str(idx + 1)]):
                    neg = random.randint(1, self.item_count)
                signal = [idx + 1, pos, neg]
            total.append(signal)
        return np.array(total)

    def __len__(self):
        return self.user_count


class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path

        self.edge_index = {}
        self.user_behaviour_degree = []
        self.behavior_adjs = {}

        self.__get_count()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict_new()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):   # 从count.txt中读取user和item的数量。
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def _normalize_adj(self, adj):
        adj = adj + sp.eye(adj.shape[0])
        degree = np.array(adj.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5),[-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)]=0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return adj.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()   # ?对代码有点问题

    def sp_mat_to_torch_sp_tensor(self, mat):
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return  torch.sparse.FloatTensor(idxs, vals, shape)

    def __get_sparse_interact_dict_new(self):
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                val = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))   # the first number is userID
                    col.append(int(line[1]))   # the second number is itemID
                    val.append(1.0)            # to imply interaction exists.
                mat = sp.coo_matrix((val, (row, col)), shape=(self.user_count + 1, self.item_count + 1))
                self.user_behaviour_degree.append(self.sp_mat_to_torch_sp_tensor(mat).to_dense().sum(axis=1).view(-1, 1))

                a = sp.csr_matrix((self.user_count + 1, self.user_count + 1))
                b = sp.csr_matrix((self.item_count + 1, self.item_count + 1))
                adj = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]) # adj: 横u i，竖u i
                # idxs = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
                # vals = torch.from_numpy(adj.data.astype(np.float32))
                # shape = torch.Size(adj.shape)

                # self.behavior_adjs[behavior] = self.sp_mat_to_torch_sp_tensor(adj)   # TODO：改了这里！加了归一化norm
                adj_temp = self._normalize_adj(adj)
                self.behavior_adjs[behavior] =  self.sp_mat_to_torch_sp_tensor(adj_temp)

                col = [x + self.user_count + 1 for x in col]
                row, col = [row, col], [col, row]
                row, col = torch.LongTensor(row).view(-1), torch.LongTensor(col).view(-1)
                all_row.append(row)
                all_col.append(col)
                self.edge_index[behavior] = torch.stack([row, col])

        self.user_behaviour_degree = torch.cat(self.user_behaviour_degree, dim=1)
        all_row = torch.cat(all_row, dim=-1)
        all_col = torch.cat(all_col, dim=-1)
        self.all_edge_index = torch.stack([all_row, all_col])

    def __get_sparse_interact_dict(self):
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))
                indices = np.vstack((row, col))
                indices = torch.LongTensor(indices)
                values = torch.ones(len(row), dtype=torch.float32)
                adj = torch.sparse.FloatTensor(indices,values, [self.user_count + 1, self.item_count + 1])
                self.user_behaviour_degree.append(adj.to_dense().sum(dim=1).view(-1, 1))
                col = [x + self.user_count + 1 for x in col]
                row, col = [row, col], [col, row]
                row = torch.LongTensor(row).view(-1)
                all_row.append(row)
                col = torch.LongTensor(col).view(-1)
                all_col.append(col)
                edge_index = torch.stack([row, col])
                self.edge_index[behavior] = edge_index
        self.user_behaviour_degree = torch.cat(self.user_behaviour_degree, dim=1)
        all_row = torch.cat(all_row, dim=-1)
        all_col = torch.cat(all_col, dim=-1)
        self.all_edge_index = torch.stack([all_row, all_col])

    def behavior_dataset(self):
        return BehaviorDate(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)

    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['cart', 'click', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='./data/Tmall', help='')
    args = parser.parse_args()
    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5)
    for index, item in enumerate(loader):
        print(index, '-----', item)
