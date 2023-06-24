# This file contains the class needed for contrastive learning 
import logging
import os
import sys

from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel
import random

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

class SimTAB(nn.Module):
    # the encoder is bert+projector
    def __init__(self, hp, device='cuda', lm='bert'):
        super().__init__()
        self.hp = hp
        self.n_classes = hp.n_classes
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768

        # projector
        sizes = [hidden_size] + list(map(int, hp.projector.split('-'))) # [768, 768]
        self.projector = nn.Linear(hidden_size, sizes[-1])

        # a fully connected layer for fine tuning
        self.fc = nn.Linear(hidden_size, self.n_classes)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().to(device)


    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        #print('logits', logits, logits.shape)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        #print('labels', labels, labels.shape)
        logits = logits / temperature
        return logits, labels


    def forward(self, y1, y2, clsA, clsB, flag=True, da=None, cutoff_ratio=0.1):
        if flag:
            # pre-training
            # encode
            batch_size = len(y1)
            y1 = y1.to(self.device) # original
            y2 = y2.to(self.device) # augment
            indexA = torch.zeros(512, clsA.shape[0])
            indexA.scatter_(0, clsA.transpose(0,1), 1)
            indexA = indexA.transpose(0,1)
            indexB = torch.zeros(512, clsB.shape[0])
            indexB.scatter_(0, clsB.transpose(0,1), 1)
            indexB = indexB.transpose(0,1)

            index = torch.cat((indexA, indexB))
            nonzero = index.nonzero()

            y = torch.cat((y1, y2))
            
            z = self.bert(y)[0][nonzero[:, 0], nonzero[:, 1]]
            z = self.projector(z)
        
            # simclr
            logits, labels = self.info_nce_loss(features=z, batch_size = z.shape[0]/2, n_views=2)
            loss = self.criterion(logits, labels)
            return loss
            
        else:
            # finetune
            x1 = y1
            x1 = x1.to(self.device) # (batch_size, seq_len)
            indexA = torch.zeros(512, clsA.shape[0])
            indexA.scatter_(0, clsA.transpose(0,1), 1)
            indexA = indexA.transpose(0,1)
            nonzero = indexA.nonzero()
            enc = self.projector(self.bert(x1)[0][nonzero[:, 0], nonzero[:, 1]]) # (batch_size, emb_size)
            out = self.fc(enc)
            return out

