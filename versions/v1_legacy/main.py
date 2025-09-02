import argparse
import random
import torch
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # data loader
    parser.add_argument('--task_name', type=str, required=False, default='classification',
                        help='task name, options:[classification, regression]')
    parser.add_argument('--root_path', type=str, default='Project1/datafiles', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='1H.csv', help='data file')
    parser.add_argument('--seq_len', type=int, default=21, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--test_size', type=float, default=0.2, help='test set ratio (0~1)')
    parser.add_argument('--threshold', type=float, default=0.6, help='correlation threshold for graph construction')
    
    # model define
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0') 
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads,多头注意力的头数')
    parser.add_argument('--factor', type=int, default=5, help='attn factor,用于 attention 中的采样因子')  
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn,FeedForward 网络的隐藏层维度')
    parser.add_argument('--activation', type=str, default='gelu', help='activation') 
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')  


    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--gcn_hidden_dim', type=int, default=128, help='GCN隐藏层维度，控制模型复杂度')
    parser.add_argument('--gcn_output_dim', type=int, default=32, help='GCN输出维度，决定最终特征表示的维度')
    
