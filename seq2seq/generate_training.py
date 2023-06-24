import os
import csv
import json
import random
import numpy as np
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

class Token_Augmenter(object):
    """Data augmentation operator.

    Support both span and attribute level augmentation operators.
    """
    def __init__(self, augment_steps):

        self.augment_steps = augment_steps
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp['bert'])
        self.max_len = 128

        ops = ['drop_token',
            'repl_token',
            'swap_token',
            'ins_token',
            'del_span',
            'shuffle_span',
            'ins_sym'
            ]

        N = self.augment_steps

        op_seq_set = []

        for num in range(len(ops)**N): 
            lst=[]
            while (num//len(ops)!=0):
                lst.append(num%len(ops))
                num=num//len(ops)
            lst.append(num%len(ops))
            a = ''.join(list(map(str,lst))[::-1])
            length_a = len(a)
            if length_a < N:
                for i in range(N-length_a):
                    a = '0'+a
            cur_op = 'best-'+a
            op_seq_set.append(cur_op)
        
        self.op_seq_set = op_seq_set

    def word_sym(self, token_id):
        synonyms = []
        candidate = self.tokenizer.decode([token_id])
        for syn in wordnet.synsets(candidate):
            for i in syn.lemmas():
                synonyms.append(i.name())
        if len(synonyms) == 0:
            return [token_id]
        else:
            return self.tokenizer.convert_tokens_to_ids(synonyms)

    def update_cur_CLS(self, tokens):
        
        indexes = [i for i, x in enumerate(tokens) if x == 101]
        return indexes 

    def compute_avg_CLS_diff(self, CLS, tokens):
        CLS += [len(tokens)-1]
        differences = np.diff(CLS)
        average = np.mean(differences)
        return average

    def augment(self, tokens, op='swap_token'):
        """ Performs data augmentation on a sequence of tokens

        The supported ops:
           ['drop_token',
            'repl_token',
            'swap_token',
            'ins_token',
            'del_span',
            'shuffle_span',
            'ins_sym'
            ]

        Args:
            tokens (list of strings): the input tokens
            op (str, optional): a string encoding of the operator to be applied

        Returns:
            list of strings: the augmented tokens
        """
        if op == 'del_span':
            if len(tokens) < 10: # in case we obtain too many empty sequences
                return tokens 
            else:
                CLS = self.update_cur_CLS(tokens)
                span_len = max(int(random.uniform(0, 0.5)*self.compute_avg_CLS_diff(CLS,tokens)),1)
                pos1, pos2 = self.sample_span(tokens, CLS, span_len=span_len)
                if pos1 < 0: # input length is zero
                    return tokens
                new_tokens = tokens[:pos1] + tokens[pos2+1:] # delete the tokens between pos1 and pos2
                return new_tokens
        elif op == 'shuffle_span':
            span_len = random.randint(2, 4)
            CLS = self.update_cur_CLS(tokens)
            pos1, pos2 = self.sample_span(tokens, CLS, span_len=span_len)
            if pos1 < 0:
                return tokens
            sub_arr = tokens[pos1:pos2+1]
            random.shuffle(sub_arr) # shuffle the tokens in between pos1 and pos2
            new_tokens = tokens[:pos1] + sub_arr + tokens[pos2+1:]
            return new_tokens
        elif op == 'drop_token':
            new_tokens = []
            for token in tokens:
                if random.randint(0, 4) != 0 or (token == 101 or token == 102):
                    new_tokens.append(token)
            return new_tokens
        elif op == 'ins_sym':
            if len(tokens) >= self.max_len:
                return tokens
            pos = self.sample_position(tokens)
            if pos == -1:
                return tokens
            symbol = random.choice(range(999,1067))
            new_tokens = tokens[:pos] + [symbol] + tokens[pos:]
            return new_tokens
        elif op == 'swap_token':
            new_tokens = tokens
            if len(new_tokens) < 4:
                return new_tokens
            CLS = self.update_cur_CLS(tokens)
            updated_CLS = CLS + [len(new_tokens) - 1]
            for i in range(len(CLS)):               
                if random.randint(0,3) == 0:
                    cur_CLS_idx = random.randint(0, len(CLS)-1)
                    if updated_CLS[cur_CLS_idx]+1 < updated_CLS[cur_CLS_idx+1]-1:
                        index1 = random.randint(updated_CLS[cur_CLS_idx]+1, updated_CLS[cur_CLS_idx+1]-1)
                        index2 = random.randint(updated_CLS[cur_CLS_idx]+1, updated_CLS[cur_CLS_idx+1]-1)
                        new_tokens[index1], new_tokens[index2] = new_tokens[index2], new_tokens[index1]

            return new_tokens
        elif op == 'repl_token':
            new_tokens = tokens
            for i in range(len(tokens)):
                if random.randint(0,4) == 0 and not (tokens[i] == 101 or tokens[i] == 102):
                    new_tokens[i] = random.choice(self.word_sym(new_tokens[i]))
            return new_tokens
        elif op == 'ins_token':
            if len(tokens) >= self.max_len:
                return tokens
            new_tokens = []
            for i in range(len(tokens)):
                new_tokens.append(tokens[i])
                if random.randint(0,1) == 0 and not (tokens[i] == 101 or tokens[i] == 102):
                    replacement = random.choice(self.word_sym(tokens[i]))
                    new_tokens.append(replacement)
            return new_tokens
        else:
            raise ValueError('DA operator not found')
            new_tokens = tokens
            return new_tokens


    def augment_command(self, tokens, op='all'):
        """ Performs data augmentation on a classification example.

        Similar to augment(tokens, labels) but works for sentences
        or sentence-pairs.

        Args:
            text (str): the input sentence
            op (str, optional): a string encoding of the operator to be applied

        Returns:
            str: the augmented sentence
        """
        if op == 'all':
            # RandAugment: https://arxiv.org/pdf/1909.13719.pdf
            N = self.augment_steps
            ops = ['drop_token',
            'repl_token',
            'swap_token',
            'ins_token',
            'del_span',
            'shuffle_span',
            'ins_sym'
            ]
            # 
            for op in random.choices(ops, k=N):
                tokens = self.augment(tokens, op=op)
        elif op == 'corrupt':
            ops = ['drop_token',
            'repl_token',
            'swap_token',
            'ins_token',
            'del_span',
            'shuffle_span',
            'ins_sym'
            ]
            for op in ops:
                tokens = self.augment(tokens, op=op)
        
        elif op[:4] == 'best':
            ops = ['drop_token',
            'repl_token',
            'swap_token',
            'ins_token',
            'del_span',
            'shuffle_span',
            'ins_sym'
            ]
            op_seq = op[5:]
            for op_idx in op_seq:
                op_idx = int(op_idx)
                tokens = self.augment(tokens, op=ops[op_idx])
            return tokens, op_seq
        else:
            tokens = self.augment(tokens, op=op)

        return tokens

    def augment_command_rl(self, tokens, epsilon, cur_policies={}):
        op_seq_set = self.op_seq_set

        if len(cur_policies.keys()) == 0:
            cur_policy = random.choice(op_seq_set)
            augmented_sent, _ = self.augment_command(tokens, op=cur_policy)
            return augmented_sent, cur_policy
        else:
            op_seq_set = set(op_seq_set)
            cur_policies = set(cur_policies.keys())
            op_seq_set = op_seq_set - cur_policies
            cur_policies = list(cur_policies)
            op_seq_set = list(op_seq_set)
            v = random.random()
            if v < epsilon:
                cur_policy = random.choice(cur_policies)
            else:
                cur_policy = random.choice(op_seq_set)
            augmented_sent, _ = self.augment_command(tokens, op=cur_policy)
            return augmented_sent, cur_policy

    def sample_span(self, tokens, CLS, span_len=3):
        candidates = []
        CLS += [len(tokens)-1]
        cls_point = set(CLS)
        for idx, token in enumerate(tokens):
            
            cut_flag = False
            if idx + span_len - 1 < len(tokens) and idx > 0 and idx < len(tokens) - 1:
                for j in range(idx, idx+span_len):
                    if j in cls_point:
                        cut_flag = True
                        break
                if cut_flag == False:
                    candidates.append((idx, idx+span_len-1))
        if len(candidates) <= 0:
            return -1, -1
        return random.choice(candidates)

    def sample_position(self, tokens):
        if len(tokens) == 0 or len(tokens) == 1:
            return -1
        elif len(tokens) == 2:
            return 1
        else:
            return random.choice(range(1, len(tokens)-1))

if __name__ == '__main__':
    DA_steps = 5

    aug = Token_Augmenter(DA_steps)

    inputs = [101, 5599, 2122, 101, 9988, 7888, 6767, 101, 3728, 8882, 10003, 102]
    for i in range(10):
        augmented = aug.augment_command_rl(inputs, 0.5)

        print(augmented)
        print(aug.tokenizer.decode(inputs))
        print(aug.tokenizer.decode(augmented[0]))
        print('=================================')
    
   
    
    