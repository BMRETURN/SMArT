import sys
sys.path.insert(0, '/share/jobdata/d0122001101428/Try')
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from tqdm import tqdm
from models.SMArT_Qwen import SMArT_Qwen
from models.SMArT_GPT2 import SMArT_GPT2

import torch
import torch.nn as nn

import os
import time

import warnings
import numpy as np

import argparse
import random

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SMArT_Qwen')

    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--root_path', type=str, default='./datasets/traffic/')
    parser.add_argument('--data_path', type=str, default='traffic.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--timestamp', type=str, default='H')

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--enc_n_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=12)
    parser.add_argument('--num_kernels', type=int, default=4)

    parser.add_argument('--loss_func', type=str, default='mse')
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--prompt', type=str, default='ETT')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)

    parser.add_argument('--itr', type=int, default=3)
    parser.add_argument('--cos', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    SEASONALITY_MAP = {
       "minutely": 1440,
       "10_minutes": 144,
       "half_hourly": 48,
       "hourly": 24,
       "daily": 7,
       "weekly": 1,
       "monthly": 12,
       "quarterly": 4,
       "yearly": 1
    }

    mses = []
    maes = []

    for ii in range(args.itr):

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.gpt_layers,
                                                                        args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = 'h'

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        time_now = time.time()
        train_steps = len(train_loader)
        model = SMArT_Qwen(args, device)
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        mse, mae = test(model, test_data, test_loader, args, device, ii)