# This file contains the training script
import os
import torch
from .simtab_inner_semtab import SimTAB
from transformers import AdamW
from apex import amp
from torch.utils import data
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import trange
from tqdm import tqdm
from .epida_scores import *
import math
import copy

import sys
from nltk.corpus import wordnet

sys.path.append('/CORDA/seq2seq/')

from generate_training_semtab import Token_Augmenter

from transformers import AutoTokenizer
import time

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

def finetune(train_iter, model, optimizer, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = torch.nn.CrossEntropyLoss()
    for i, batch in enumerate(tqdm(train_iter)):
        optimizer.zero_grad()
        x, y, _, clsA = batch
        prediction = model(x, None, clsA, None, flag=False)
        loss = criterion(prediction, y.to(model.device))
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if i % 200 == 0: # monitoring
            print(f"    fine tune step: {i}, loss: {loss.item()}")
        del loss

def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def evaluation(model, iter, model_save_path, is_test = False, cur_best_loss=100):
    if is_test:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
    labels = []
    predicted_labels = []
    loss_fn = torch.nn.CrossEntropyLoss()
    v_epoch_loss = 0
    acc_time = 0
    for i, item in enumerate(iter):
        x, y, _, clsA = item
        start = time.time()
        prediction = model(x, None, clsA, None, flag=False)
        end = time.time()
        acc_time = acc_time + end - start
        vloss = loss_fn(prediction, y.to(model.device))
        v_epoch_loss += vloss.item()
        predicted_label = torch.argmax(prediction, dim=1)
        labels.append(y.detach().numpy())
        predicted_labels += predicted_label.detach().cpu().numpy().tolist()
        del x, vloss
    v_length_label = len(labels)
    v_total_loss = v_epoch_loss / (v_length_label)
    print("loss:", v_total_loss)
    print('accu time', acc_time, len(iter), 'avg time', acc_time/len(iter))
    
    if v_total_loss < cur_best_loss:
        cur_best_loss = v_total_loss
        if not is_test:
            torch.save(model.state_dict(), model_save_path)
            print('model updated')
    
    pred_labels = np.concatenate([np.expand_dims(i, axis=0) for i in predicted_labels]) 
    true_labels = np.concatenate(labels)
    f1_scores = metric_fn(pred_labels, true_labels)
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1'], cur_best_loss

def train(trainset_nolabel, train_set, valid_set, test_set, hp):
    print('=====================================================================')
    print('start training')
    print('lr ', hp.lr, ' f_lr', hp.f_lr)
    model_save_path = hp.save_path
    num_ssl_epochs = hp.n_ssl_epochs

    train_nolabel_iter = data.DataLoader(dataset=trainset_nolabel,
                                             batch_size=hp.contrastive_batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=trainset_nolabel.pad
                                        )
    
    train_iter = data.DataLoader(dataset=train_set,
                                    batch_size=hp.batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    collate_fn=train_set.pad)
    
    valid_iter = data.DataLoader(dataset=valid_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=valid_set.pad)
    
    test_iter = data.DataLoader(dataset=test_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=test_set.pad)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimTAB(hp, device=device, lm=hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    optimizer_f = AdamW(model.parameters(), lr=hp.f_lr) 
    if hp.fp16:
        opt_level = '02'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    
    cur_best = 100
    
    if hp.switching == False:
        for epoch in range(hp.n_epochs):
            if epoch < num_ssl_epochs:
                for i, batch in enumerate(tqdm(train_nolabel_iter)):
                    yA, yB, clsA, clsB = batch
                    optimizer.zero_grad()
                    loss = model(yA, yB, clsA, clsB, da=hp.da)
                    if hp.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    if i % 200 == 0: # monitoring
                        print(f"    step: {i}, loss: {loss.item()}")
                    del loss
            else:
                model.train()
                finetune(train_iter, model, optimizer_f, hp)
                print('start validating epoch: ', epoch-num_ssl_epochs)
                model.eval()
                v_sw, v_ma, cur_best_loss = evaluation(model, valid_iter, model_save_path, is_test = False, cur_best_loss=cur_best)
                cur_best = cur_best_loss
                cur_best_model = model
    else:
        cur_ssl = 1
        for epoch in range(hp.n_epochs):
            cur_epoch = epoch + 1
            if cur_epoch % 2 == 1 and cur_ssl <= num_ssl_epochs:
                for i, batch in enumerate(tqdm(train_nolabel_iter)):
                    yA, yB, clsA, clsB = batch
                    optimizer.zero_grad()
                    loss = model(yA, yB, clsA, clsB, da=hp.da)
                    if hp.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    if i % 10 == 0: # monitoring
                        print(f"    step: {i}, loss: {loss.item()}")
                    del loss
                cur_ssl += 1
            else:
                
                model.train()
                finetune(train_iter, model, optimizer_f, hp)
                print('start validating epoch: ', epoch-cur_ssl+1)
                model.eval()
                v_sw, v_ma, cur_best_loss = evaluation(model, valid_iter, model_save_path, is_test = False, cur_best_loss=cur_best)
                cur_best = cur_best_loss
                cur_best_model = model

    print('start testing after ssl before epida')
    model = cur_best_model
    model.eval()
    t_sw, t_ma, t_loss = evaluation(model, test_iter, model_save_path, is_test = True, cur_best_loss=100)


def generate_string(input_string):
    str_ = ""
    for idx, item in enumerate(input_string):
        if not idx == 0:
            str_ += ' '
        str_ += item
    return str_

def epida_sent_rl(base_col, label, hp, model, augmenter, tokenizer, policy_set, epsilon, cls_idx): # generate and select the high quality content augmentation
    da_size = hp.da_size
    amp_ratio = hp.amplification
    
    augment_content = []
    content_policy = []
    for _ in range(da_size*amp_ratio):
        cur_aug_content, cur_policy = augmenter.augment_command_rl(base_col, epsilon, policy_set)
        augment_content.append(cur_aug_content)
        content_policy.append(cur_policy)
    augment_content = [base_col] + augment_content

    labels = []
    CLSs = []
    tokenized_content = augment_content

    for aug_c in augment_content:
        labels.append(label)
        CLSs.append(augmenter.update_out_CLS(aug_c, cls_idx))
    maxlen = max([len(x) for x in tokenized_content])

    old_tokenized_content = [xi + [0]*(maxlen - len(xi)) for xi in tokenized_content]
    tokenized_content = torch.LongTensor(old_tokenized_content)
    inputs = tokenized_content.cuda()
    new_CLSs = torch.LongTensor(CLSs)

    outputs = model(inputs, None, new_CLSs, None, flag = False)

    total_scores = []

    num_of_tables = tokenized_content.shape[0]
    num_of_columns = outputs.shape[0]
    columns_per_table = int(num_of_columns/num_of_tables)

    for j in range(columns_per_table):
        ups, downs, scores = [], [], []
        for i in range(0, num_of_tables):
            b = torch.softmax(outputs[i*columns_per_table+j],0)
            c = torch.softmax(outputs[j],0)
            C = b.size(0)
            a = torch.zeros(C).cuda()
            a[label] = 1.0
            _up,_down = gradmutualgain(outputs[i*columns_per_table+j],a,b,c,loss_fn=torch.nn.CrossEntropyLoss())
            ups.append(_up.item())
            downs.append(_down.item())
        ups = np.array(ups)
        downs = np.array(downs)
        ups = (ups-np.min(ups))/(np.max(ups)-np.min(ups))
        downs = (downs-np.min(downs))/(np.max(downs)-np.min(downs))
        alpha_epda = 0.5
        for i in range(downs.shape[0]):
            _up,_down=ups[i],downs[i]
            score = alpha_epda * _up + (1.0-alpha_epda)*_down
            if score == np.nan or math.isnan(score):
                score = 1.0
            scores.append(score)
        scores = np.array(scores)
        total_scores.append(scores)
    
    total_scores = np.mean(total_scores,0)
    sortargs = np.argsort(-total_scores).tolist()

    new_content = []
    out_CLSs = []
    out_cls_idx = []

    cls_idx = cls_idx.tolist()[0]
    new_content.append(old_tokenized_content[0])
    out_CLSs.append(CLSs[0])
    out_cls_idx.append(cls_idx)

    for idx in sortargs[:da_size]:
        if not idx == 0:    
            new_content.append(old_tokenized_content[idx])
            out_CLSs.append(CLSs[idx])
            out_cls_idx.append(cls_idx)
            if content_policy[idx-1] in policy_set.keys():
                policy_set[content_policy[idx-1]] = total_scores[idx]
            elif len(policy_set.keys()) < da_size-1:
                policy_set[content_policy[idx-1]] = total_scores[idx]
            else:
                replaced_key = min(policy_set, key=policy_set.get)
                del policy_set[replaced_key]
                policy_set[content_policy[idx-1]] = total_scores[idx]
    return new_content, policy_set, out_CLSs, out_cls_idx

def update_dataloader_sent(train_iter, hp, model_save_path, epsilon, policy_set):
    Xs, ys, cls_idxs, clss = [], [], [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimTAB(hp, device=device, lm=hp.lm)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    model = model.cuda()
    da_steps = hp.da_steps
    augmenter = Token_Augmenter(da_steps)
    tokenizer = AutoTokenizer.from_pretrained(lm_mp['bert'])
    for i, batch in enumerate(tqdm(train_iter)):

        x, y, cls_idx, _ = batch
        x = x[0].tolist()
        augmented_content, policy_set, cls_set, cls_idx_set = epida_sent_rl(x, y, hp, model, augmenter, tokenizer, policy_set, epsilon, cls_idx)
        Xs += augmented_content
        ys += [y for i in range(len(augmented_content))]
        clss += cls_set
        cls_idxs += cls_idx_set
    return Xs, ys, clss, cls_idxs, policy_set


def finetune_epida_rl(train_set, valid_set, test_set, hp, epsilon):
    print('=====================================================================')
    print('start finetuning on epida')
    print(' e_lr', hp.e_lr)
    model_save_path = hp.save_path
    updated_train_set = copy.deepcopy(train_set)
    train_iter_in = data.DataLoader(dataset=train_set,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=0,
                                    collate_fn=train_set.pad)

    train_iter = data.DataLoader(dataset=updated_train_set,
                                    batch_size=hp.batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    collate_fn=updated_train_set.pad)
    
    valid_iter = data.DataLoader(dataset=valid_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=valid_set.pad)
    
    test_iter = data.DataLoader(dataset=test_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=test_set.pad)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimTAB(hp, device=device, lm=hp.lm)

    if not os.path.exists(model_save_path):
        torch.save(model.state_dict(), model_save_path)
    
    model.load_state_dict(torch.load(model_save_path))
    

    optimizer = AdamW(model.parameters(), lr=hp.e_lr)
    # scheduler空缺
    if hp.fp16:
        opt_level = '02'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    
    model = model.cuda()

    cur_best = 100
    
    criterion = torch.nn.CrossEntropyLoss()

    policy_set = {}
    for epoch in range(hp.n_epida_epochs):
        xs, ys, clss, cls_idxs, policy_set = update_dataloader_sent(train_iter_in, hp, model_save_path, epsilon, policy_set)
        train_iter.dataset.update(xs, ys, cls_idxs, clss)
        
        model.train()
        for i, batch in enumerate(tqdm(train_iter)):
            x, y, _, clsA = batch
            optimizer.zero_grad()
            prediction = model(x, None, clsA, None, flag=False)
            loss = criterion(prediction, y.to(model.device))
            if hp.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            if i % 200 == 0: # monitoring
                print(f"    fine tune step: {i}, loss: {loss.item()}")
            del loss

        print('start validating epida epoch: ', epoch)
        model.eval()
        v_sw, v_ma, cur_best_loss = evaluation(model, valid_iter, model_save_path, is_test = False, cur_best_loss=cur_best)
        cur_best = cur_best_loss
        cur_best_model = model
    
    print('start testing after epida')
    model = cur_best_model
    model.eval()
    t_sw, t_ma, t_loss = evaluation(model, test_iter, model_save_path, is_test = True, cur_best_loss=100)
