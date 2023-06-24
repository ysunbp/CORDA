from functools import reduce
import operator
import os
import csv
import numpy as np
import json
import torch
import random
from torch.utils import data
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import jsonlines
import itertools


lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

class SADataset(data.Dataset): #copy from BTDataset class
    """Dataset for pre-training"""

    def __init__(self,
                    csv_file_dir,
                    max_length=128,
                    size=None,
                    lm='bert',
                    da='all'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_length = max_length
        self.size = size # dataset size
        jsonl_dir = './sato_data/rels'
        pd_list = []

        folds = [0,1,2,3,4]
        rel_dict = {}
        for fold in tqdm(folds):
            jsonl_path = jsonl_dir + str(fold)+'.jsonlines'
            csv_file_path = csv_file_dir+str(fold)+'.csv'
            cur_df = pd.read_csv(csv_file_path)
            pd_list.append(cur_df)
            with jsonlines.open(jsonl_path) as f:
                for line in f:
                    #print(dict(line))
                    rel_dict.update(dict(line))
        
        merged_dataset = pd.concat(pd_list, axis = 0)
        data_list = []
        data_dict = {}

        for i, (index, group_df) in enumerate(tqdm(merged_dataset.groupby("table_id"))):
            index = index[24:]
            rel_names = rel_dict[index]
            if len(rel_names) == 0: # sample related tables
                rel_flag = 0
            else:
                rel_flag = 1
            
            rel_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(self.shuffle_col(x), add_special_tokens=True, max_length=max_length, truncation=True))
            rel_ids_up = reduce(operator.add, rel_ids_list)
            
            token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True))
            token_ids_up = reduce(operator.add, token_ids_list)
            
            if len(token_ids_up) > 512:
                token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(
                x, add_special_tokens=True, max_length=int(max_length/2), truncation=True)).tolist(
                )
                token_ids_up = reduce(operator.add,
                                                token_ids_list)
            token_ids = token_ids_up
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]

            for cls_index in cls_index_list:
                assert token_ids_up[
                    cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = cls_index_list

            class_ids = group_df["class_id"].values

            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes, rel_flag, rel_ids_up, cls_indexes])
            data_dict[index] = [token_ids, cls_indexes]
        
        for i, cur_list in enumerate(tqdm(data_list)):
            if cur_list[5] == 0:
                continue
            else:
                candidate_rels = rel_dict[cur_list[0]]
                selected_neighbor = random.choice(candidate_rels)
                try:
                    identified_data_tensor = data_dict[selected_neighbor][0]
                    identified_cls = data_dict[selected_neighbor][1]
                except KeyError:
                    print('error', cur_list[0], candidate_rels)
                    continue
                data_list[i][6] = identified_data_tensor
                data_list[i][7] = identified_cls

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "rel_flag", "rel_tensor", "rel_cls"
                                     ])        

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.table_df)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

            Args:
                idx (int): the index of the item

            Returns:
                List of int: token ID's of the 1st entity
                List of int: token ID's of the 2nd entity
                List of int: token ID's of the two entities combined
                int: the label of the pair (0: unmatch, 1: match)
        """
        return self.table_df.iloc[idx]["data_tensor"], self.table_df.iloc[idx]["rel_tensor"], self.table_df.iloc[idx]["cls_indexes"], self.table_df.iloc[idx]["rel_cls"], 

    def shuffle_col(self, col_str):
        col_cells = col_str.split()
        random.shuffle(col_cells)
        out_str = ''.join(col_cells)
        return out_str

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

            Args:
                batch (list of tuple): a list of dataset items

            Returns:
                LongTensor: x1 of shape (batch_size, seq_len)
                LongTensor: x2 of shape (batch_size, seq_len).
                            Elements of x1 and x2 are padded to the same length
        """
        yA, yB, clsA, clsB = zip(*batch)

        maxlen = max([len(x) for x in yA])
        maxlen = max(maxlen,max([len(x) for x in yB]))
        maxlen2 = max([len(c1) for c1 in clsA])

        yA = [xi + [0]*(maxlen - len(xi)) for xi in yA]
        yB = [xi + [0]*(maxlen - len(xi)) for xi in yB]
        clsA = [ci + [0]*(maxlen2 - len(ci)) for ci in clsA]
        clsB = [ci + [0]*(maxlen2 - len(ci)) for ci in clsB]

        return torch.LongTensor(yA), torch.LongTensor(yB), torch.LongTensor(clsA), torch.LongTensor(clsB)


class SupSADataset(data.Dataset):
    """dataset for the evaluation"""

    def __init__(self,
                    csv_file_dir, folds,
                    max_length=128,
                    size=None,
                    lm='bert',
                    da='all'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_length = max_length
        self.size = size # dataset size
        pd_list = []

        rel_dict = {}
        for fold in tqdm(folds):
            csv_file_path = csv_file_dir+str(fold)+'.csv'
            cur_df = pd.read_csv(csv_file_path)
            pd_list.append(cur_df)
        
        merged_dataset = pd.concat(pd_list, axis = 0)
        data_list = []
        data_dict = {}

        for i, (index, group_df) in enumerate(tqdm(merged_dataset.groupby("table_id"))):
            index = index[24:]
            
            token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True))
            token_ids_up = reduce(operator.add, token_ids_list)

            if len(token_ids_up) > 512:
                token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(
                x, add_special_tokens=True, max_length=int(max_length/2), truncation=True)).tolist(
                )
                token_ids_up = reduce(operator.add,
                                                token_ids_list)
            token_ids = token_ids_up
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]

            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = cls_index_list
            class_ids = list(group_df["class_id"].values)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
            data_dict[index] = [token_ids, cls_indexes]

        if size is not None:
            if size > len(data_list):
                N = size // len(data_list) + 1
                data_list = (data_list * N)[:size] # over sampling
            else:
                indices = [i for i in range(len(data_list))]
                selected_indices = random.sample(indices, size)
                out_instances = []
                for index in selected_indices:
                    out_instances.append(data_list[index])
                data_list = out_instances

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])        

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.table_df)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

            Args:
                idx (int): the index of the item

            Returns:
                List of int: token ID's of the 1st entity
                List of int: token ID's of the 2nd entity
                List of int: token ID's of the two entities combined
                int: the label of the pair (0: unmatch, 1: match)
        """
        return self.table_df.iloc[idx]["data_tensor"], self.table_df.iloc[idx]["label_tensor"], self.table_df.iloc[idx]["cls_indexes"]

    def shuffle_col(self, col_str):
        col_cells = col_str.split()
        random.shuffle(col_cells)
        out_str = ''.join(col_cells)
        return out_str

    def update(self, Xs, ys, clss): 
        result = [[subXs, subys, subclss] for subXs, subys, subclss in zip(Xs, ys, clss)]
        data_tensor = pd.DataFrame(result, columns=['data_tensor','label_tensor','cls_indexes'])
        self.table_df = data_tensor
        return

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: a batch of labels, (batch_size,)
        """
        # cleaning
        x1, y, clsA = zip(*batch)
        maxlen1 = max([len(x) for x in x1])
        maxlen2 = max([len(c1) for c1 in clsA])
        out_y = []
        x1 = [xi + [0]*(maxlen1 - len(xi)) for xi in x1]
        for yi in y:
            out_y += yi
        clsA = [ci + [0]*(maxlen2 - len(ci)) for ci in clsA] #fill with 0, will be counter off by cls 0 in scatter_ function since all the columns are annotated
        
        return torch.LongTensor(x1), torch.LongTensor(out_y), torch.LongTensor(clsA)

if __name__ == '__main__':
    lm = 'bert'
    max_len = 128
    da = 'empty'
    SupSADataset('./sato_data/msato_cv_', folds=[0],
                                    lm=lm,
                                    size=None,
                                    max_length=max_len,
                                    da=da) # data augmentation