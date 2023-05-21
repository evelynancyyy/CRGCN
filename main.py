#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: yanms
# @Date  : 2021/11/1 15:25
# @Desc  :
import argparse
import random
import time
import os
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from data_set import DataSet

from model_cascade import CRGCN

from trainer import Trainer

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--node_dropout', type=float, default=0.75)
    parser.add_argument('--message_dropout', type=float, default=0.25)

    parser.add_argument('--data_name', type=str, default='beibei', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')

    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.003, help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=int, default=500, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--gpu', type=int, default=0, help='')

    parser.add_argument('--keep_rate', type=float, default=0.9, help='')
    parser.add_argument('--if_multi_tasks', action='store_true', help='')

    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'Tmall'
    elif args.data_name == 'tmall_cold':
        args.data_path = 'data/Tmall_cold_all'
        args.behaviors = ['click', 'cart', 'collect', 'buy']
        args.model_name = 'Tmall_cold_all'
    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        args.behaviors = ['click', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'beibei'
    elif args.data_name == 'beibei_cold':
        args.data_path = './data/beibei_cold_all'
        args.behaviors = ['view', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'beibei'
    elif args.data_name == 'jdata':
        args.data_path = './data/jdata'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'jdata'
    else:
        raise Exception('data_name cannot be None')

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME
    logfile = '{}_enb_{}_{}'.format(args.model_name, args.embedding_size, TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)  # DONE
    model = CRGCN(args, dataset)
    model = model.to(args.device)

    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    # trainer.evaluate(0, 12, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info('train end total cost time: {}'.format(time.time() - start))



