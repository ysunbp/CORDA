import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import json
import sys
import numpy as np
import random
import torch
import mlflow
from scripts.inner_context_dataset import SADataset, SupSADataset

from torch.utils import data
from scripts.train_epida_rl_inner import train, finetune_epida_rl

csv_path = './sato_data/msato_cv_'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    epsilons = [0.75]
    bs = 8
    lr_1 = 5e-6
    lr_2 = 1e-5
    for fold in range(5):
        for epsilon in epsilons:
            setup_seed(3407)
            parser = argparse.ArgumentParser()
            parser.add_argument("--max_len", type=int, default=128)
            parser.add_argument("--lm", type=str, default='bert') # bert
            parser.add_argument("--da", type=str, default='empty')
            parser.add_argument("--da_steps", type=int, default=5)
            parser.add_argument("--da_size", type=int, default=5)
            parser.add_argument("--amplification", type=int, default=3)
            parser.add_argument("--batch_size", type=int, default=bs)
            parser.add_argument("--contrastive_batch_size", type=int, default=8)
            parser.add_argument("--lr", type=float, default=lr_1)
                
            parser.add_argument("--fp16", dest="fp16", action="store_true")
            parser.add_argument('--projector', default='768', type=str,
                                    metavar='MLP', help='projector MLP')
            parser.add_argument("--n_classes", type=int, default=78)
            parser.add_argument("--n_epochs", type=int, default=6) # default 10
            parser.add_argument("--clustering", type=bool, default=True)
            parser.add_argument("--n_ssl_epochs", type=int, default=5) # default 5
            parser.add_argument("--switching", type=bool, default=False) # switch training
            parser.add_argument("--n_epida_epochs", type=int, default=10)
            parser.add_argument("--f_lr", type=float, default=lr_2)
            parser.add_argument("--e_lr", type=float, default=lr_2)
            parser.add_argument("--save_path", type=str, default='./sato_checkpoints/5fold-full-set'+":"+str(lr_1)+':'+str(lr_2)+':'+str(lr_2)+':'+str(bs)+':'+str(bs)+'fold'+str(fold)+'.pkl')

            hp = parser.parse_args()
            print(hp.save_path)
            trainset_nolabel = SADataset(csv_path,
                                lm=hp.lm,
                                size=None,
                                max_length=hp.max_len,
                                da=hp.da) # data augmentation
            train_set = SupSADataset(csv_path, folds=[fold%5, (fold+1)%5, (fold+2)%5, (fold+3)%5], max_length=hp.max_len,
                    size=None, 
                    lm=hp.lm,
                    da=None)
            #1000
            valid_set = SupSADataset(csv_path, folds=[(fold+4)%5], max_length=hp.max_len,
                                size=None,
                                lm=hp.lm,
                                da=None)
            #500
            test_set = SupSADataset(csv_path, folds=[(fold+4)%5], max_length=hp.max_len,
                                size=None,
                                lm=hp.lm,
                                da=None)
            #500
            # TODO: change the size of datasets into a hyper-parameter
            train(trainset_nolabel, train_set, valid_set, test_set, hp)
            finetune_epida_rl(train_set, valid_set, test_set, hp, epsilon)
            print('finish checking fold', fold)
