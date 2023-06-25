import csv
import json
import os
import spacy
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import random
import jsonlines
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import operator
import time
from tqdm import trange



def read_csv(csv_dir):
    csv_lines = csv.reader(open(csv_dir))
    content = []
    num_row = 0
    for i, line in enumerate(csv_lines):
        if i == 0:
            header = line
            num_col = len(header)
            
        else:
            content.append(line)
            num_row += 1
    num_cell = num_col*num_row
    return header, content, num_col, num_row, num_cell

rounds = [1,3,4]
  
for round in rounds:
    csv_reader = csv.reader(open("./raw_data/Round"+str(round)+"/gt/CTA_Round"+str(round)+"_gt.csv"))
    csv_dir = "./raw_data/Round"+str(round)+"/tables/"
    output_dir = "./json/Round"+str(round)+"/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i = 0
    for line in csv_reader:
        file_name = line[0]+".csv"
        target_col = line[1]
        label = line[2].split('/')[-1]
        file_dir = csv_dir+file_name
        if not os.path.exists(file_dir):
            continue
        header, content, num_col, num_row, num_cell = read_csv(file_dir)
        dict = {}
        dict['filename'] = file_name
        dict['header'] = header
        dict['content'] = content
        dict['target'] = target_col
        dict['label'] = label
        output_path = output_dir+str(i)+'.json'
        with open(output_path, "w") as outfile:
            json.dump(dict, outfile)
        i += 1
        print(i)

nlp = spacy.load('en_core_web_trf')
table_dir = './json/Round'
out_dir = './json/Round'
rounds = [1,3,4]

for round in rounds:
    json_dir = table_dir+str(round)
    out_json_dir = out_dir+str(round)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    for json_file in os.listdir(json_dir):
        json_file_path = json_dir+'/'+json_file
        out_json_file_path = out_json_dir+'/'+json_file
        with open(json_file_path, 'r') as load_f:
            load_dict = json.load(load_f)
            new_dict = {}
            new_dict['filename'] = load_dict['filename']
            new_dict['header'] = load_dict['header']
            new_dict['content'] = load_dict['content']
            new_dict['target'] = load_dict['target']
            new_dict['label'] = load_dict['label']

            table_content = np.array(load_dict['content'])
            width = len(load_dict['header'])
            col_ne = []
            most_common = []
            for col_index in range(width):
                column_string = ''
                cur_ne = []
                for cell in table_content[:,col_index]:
                    column_string += cell
                    column_string += ' ; '
                doc = nlp(column_string)
                for ent in doc.ents:
                    cur_ne.append(ent.label_)
                col_ne.append(cur_ne)
                if cur_ne:
                    most_common.append(Counter(cur_ne).most_common(1)[0][0])
                else:
                    most_common.append('EMPTY')
            new_dict['table_NE'] = col_ne
            new_dict['most_common'] = most_common
            with open(out_json_file_path, 'w') as out_f:
                json.dump(new_dict, out_f)
            print(json_file)


import json
import os
import numpy as np
from tqdm import tqdm
import re


def containenglish(string): # check if the string contains any English words
    return bool(re.search('[a-zA-Z]', string))

def match_quantity(datestr): 
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    events_dict = ['Prix','Olympics','Championships', 'Open', 'Challenger', 'Trophy', 'Tournament']
    flag = 0
    for char in datestr:
        if char in numbers:
            flag = 1
            break
    if flag == 0:
        data_type = 'WORK_OF_ART'
    else:
        flag2 = 0
        for word in datestr.split(' '):
            if word in events_dict:
                flag2 = 1
                break
        if flag2 == 1:
            data_type = 'EVENT'
        else:
            data_type = 'QUANTITY'
    return data_type

def match_date(datestr): # Divide the DATE
    numbers = ['0','1','2','3','4','5','6','7','8','9']
    events_dict = ['Prix','Olympics','Championships', 'Open', 'Challenger', 'Trophy', 'Tournament']
    if datestr.isdigit(): #YYYY
        data_type = 'DATE1'
    else:
        if containenglish(datestr): #Jan
            flag = 0
            for char in datestr:
                if char in numbers:
                    flag = 1
                    break
            if flag == 0:
                data_type = 'WORK_OF_ART'    
            else:
                flag2 = 0
                for word in datestr.split(' '):
                    if word in events_dict:
                        flag2 = 1
                        break
                if flag2 == 1:
                    data_type = 'EVENT'
                else:
                    data_type = 'DATE2'
        else:
            splitted = datestr.split('-')
            if len(splitted) == 3: #YYYY-MM-DD
                data_type = 'DATE3'
            elif len(splitted) == 2: #MM-DD
                data_type = 'DATE4'
            else:
                data_type = 'DATE5'
    return data_type

def match_name(namestr): #Divide PERSON
    if '.' in namestr:
        data_type = 'PERSON1'
    else:
        data_type = 'PERSON2'
    return data_type


def preprocess_date_name(): # Process PERSON and DATE
    json_summary_base = './json/Round'
    out_base = './json/Round'
    rounds = [1,3,4]
    print('Now process the dates, quantities and names format')
    for round in rounds:
        json_summary_dir = json_summary_base+str(round)
        out_dir = out_base+str(round)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for json_file_name in tqdm(os.listdir(json_summary_dir)):
            json_summary_path = json_summary_dir+'/'+json_file_name
            out_path = out_dir+'/'+json_file_name
            with open(json_summary_path, 'r') as jf:
                content = json.load(jf)
            col_types = content['most_common']
            table_content = content['content']
            out_json = content
            for i, data_type in enumerate(col_types):
                if data_type == 'DATE':
                    candidate_str = np.array(content['content'])[0, i]
                    if not candidate_str:
                        candidate_str = np.array(content['content'])[1, i]
                    out_json['most_common'][i] = match_date(candidate_str)
                if data_type == 'PERSON':
                    candidate_str = np.array(content['content'])[0, i]
                    if not candidate_str:
                        candidate_str = np.array(content['content'])[1, i]
                    out_json['most_common'][i] = match_name(candidate_str)
                if data_type == 'QUANTITY':
                    candidate_str = np.array(content['content'])[0, i]
                    if not candidate_str:
                        candidate_str = np.array(content['content'])[1, i]
                    out_json['most_common'][i] = match_quantity(candidate_str)
            with open(out_path, 'w') as of:
                json.dump(out_json, of)

def find_related(): # Align related tables
    json_dir = './json/Round'
    out_dir = './json/Round'
    rounds = [1,3,4]
    pattern_dict = {}
    print('Now finding related tables')
    for round in rounds:
        json_path = json_dir+str(round)
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        for json_file in os.listdir(json_path):
            json_file_path = json_path+'/'+json_file
            with open(json_file_path, 'r') as load_f:
                load_dict = json.load(load_f)
            most_common_key = str(load_dict['most_common'])
            json_id = str(round)+'-'+json_file
            if most_common_key in pattern_dict.keys():
                pattern_dict[most_common_key].append(json_id)
            else:
                pattern_dict[most_common_key] = [json_id]
    for key_pattern in pattern_dict.keys():
        if len(pattern_dict[key_pattern]) == 1:
            file_name = pattern_dict[key_pattern][0]
            round, file_id = file_name.split('-')
            file_path = json_dir+round+'/'+file_id
            with open(file_path, 'r') as load_file:
                load_dict = json.load(load_file)
            out_path = out_dir+round+'/'+file_id
            related_table = [load_dict['filename']]
            load_dict['related_table'] = related_table
            with open(out_path, 'w') as dump:
                json.dump(load_dict, dump)
        else:
            table_ids = []
            for file_name in pattern_dict[key_pattern]:
                round, file_id = file_name.split('-')
                file_path = json_dir+round+'/'+file_id
                with open(file_path, 'r') as load_file:
                    load_dict = json.load(load_file)
                if load_dict['filename'] in table_ids:
                    continue
                else:
                    table_ids.append(load_dict['filename'])
            if len(table_ids) == 1:
                for file_name in pattern_dict[key_pattern]:
                    round, file_id = file_name.split('-')
                    file_path = json_dir+round+'/'+file_id
                    with open(file_path, 'r') as load_file:
                        load_dict = json.load(load_file)
                    load_dict['related_table'] = [load_dict['filename']]
                    out_path = out_dir+round+'/'+file_id
                    with open(out_path, 'w') as dump:
                        json.dump(load_dict, dump)
            else:
                for file_name in pattern_dict[key_pattern]:
                    round, file_id = file_name.split('-')
                    file_path = json_dir+round+'/'+file_id
                    with open(file_path, 'r') as load_file:
                        load_dict = json.load(load_file)
                    current_table_id = [load_dict['filename']]
                    for item in table_ids:
                        if not item in current_table_id:
                            current_table_id.append(item)
                    out_path = out_dir+round+'/'+file_id
                    out_dict = load_dict
                    out_dict['related_table'] = current_table_id
                    with open(out_path, 'w') as dump:
                        json.dump(out_dict, dump)

preprocess_date_name()
find_related()



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

pathway1 = "./json/Round1/"
pathway3 = "./json/Round3/"
pathway4 = "./json/Round4/"

train_jsonl = "./file2.jsonl"
test_jsonl = "./file1.jsonl"

csv_dir_1 = "./raw_data/Round1/tables/"
csv_dir_3 = "./raw_data/Round3/tables/"
csv_dir_4 = "./raw_data/Round4/tables/"

csv_dirs = [csv_dir_1, csv_dir_3, csv_dir_4]
pathways = [pathway1, pathway3, pathway4]


json_files = []
labels = []

# Exact alignment
for i, pathway in enumerate(pathways):
    for file_name in tqdm(os.listdir(pathway)):
        file_path = pathway + file_name
        rel_json_path = pathway + file_name
        with open(file_path, "r") as load_f:
            load_dict = json.load(load_f)
        with open(rel_json_path, "r") as load_j:
            load_dict_j = json.load(load_j)
        load_dict['table_NE'] = load_dict_j['table_NE']
        load_dict['most_common'] = load_dict_j['most_common']
        load_dict['related_table'] = load_dict_j['related_table']
        related_cols = []
        for rel_table in load_dict_j['related_table']:
            for csv_dir in csv_dirs:
                if os.path.exists(csv_dir+rel_table):
                    list_file = []
                    with open(csv_dir+rel_table,'r') as csv_file: 
                        all_lines=csv.reader(csv_file)  
                        for one_line in all_lines:  
                            list_file.append(one_line)  
                    list_file.remove(list_file[0])
                    arr_file = np.array(list_file)
                    related_cols.append(list(arr_file[:, int(load_dict['target'])]))
        load_dict['related_cols'] = related_cols
        json_files.append(load_dict)
        labels.append(load_dict['label'])


sfolder_test = StratifiedKFold(n_splits=10, random_state = 42, shuffle=True)
train_valid_set = []
test_set = []
for train_valid, test in sfolder_test.split(json_files, labels):
    train_valid_index = train_valid
    test_index = test
    break
for index in train_valid_index:
    with jsonlines.open(train_jsonl, "a") as writer:
        writer.write(json_files[index])
for index in test_index:
    with jsonlines.open(test_jsonl, "a") as writer:
        writer.write(json_files[index])


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1)+len(list2)) - intersection
    return float(intersection)/union

def read_tables(path):
    csv_reader = csv.reader(open(path))
    cur_set = []
    for i, line in enumerate(csv_reader):
        if i > 0:
            cur_set += line
    return list(set(cur_set))

def compute_jaccard(pathways):
    table_content = {}
    jaccard_dict = {}
    for pathway in pathways:
        for file_name in os.listdir(pathway):
            file_path = pathway + file_name
            table_content[file_name] = read_tables(file_path)
    total_length = len(table_content)
    for i in trange(total_length):
        list1_key = list(table_content.keys())[i]
        list1 = table_content[list1_key]
        for j in range(i+1,total_length):
            list2_key = list(table_content.keys())[j]
            list2 = table_content[list2_key]
            jaccard_value = jaccard(list1, list2)
            if jaccard_value > 0.1:
                if not list2_key in jaccard_dict.keys():
                    jaccard_dict[list2_key] = [(list1_key, jaccard_value)]
                else:
                    jaccard_dict[list2_key].append((list1_key, jaccard_value))
                if not list1_key in jaccard_dict.keys():
                    jaccard_dict[list1_key] = [(list2_key, jaccard_value)]
                else:
                    jaccard_dict[list1_key].append((list2_key, jaccard_value))
        if not list1_key in jaccard_dict.keys():
            jaccard_dict[list1_key] = []
            
    sorted_jaccard_dict = {}
    for key in jaccard_dict.keys():
        sorted_jaccard_dict[key] = sorted(jaccard_dict[key], key=lambda item:item[1], reverse=True)
    return sorted_jaccard_dict


pathways = ["./raw_data/Round1/tables/", "./raw_data/Round3/tables/", "./raw_data/Round4/tables/"]

jaccard_dict = compute_jaccard(pathways)
updated_jaccard_dict = {}
for key in jaccard_dict.keys():
    updated_jaccard_dict[key] = []
    for value in jaccard_dict[key]:
        updated_jaccard_dict[key].append(value[0])



outpath1 = './file1.jsonl'
outpath2 = './file2.jsonl'

out_test = []
out_train = []


with open('./file2.jsonl', 'r+', encoding='utf8') as f:
    for item in jsonlines.Reader(f):
        current_dict = item
        cur_filename = current_dict['filename']
        jaccard_set = updated_jaccard_dict[cur_filename]
        remained_related = []
        remained_rel_col = []
        for item in updated_jaccard_dict[cur_filename]:
            if not item in current_dict['related_table']:
                continue
            else:
                i = current_dict['related_table'].index(item)
                remained_related.append(current_dict['related_table'][i])
                remained_rel_col.append(current_dict['related_cols'][i])
        updated_dict = current_dict
        updated_dict['related_table'] = remained_related
        updated_dict['related_cols'] = remained_rel_col
        out_train.append(updated_dict)

with open('./file1.jsonl', 'r+', encoding='utf8') as f:
    for item in jsonlines.Reader(f):
        
        current_dict = item
        cur_filename = current_dict['filename']
        jaccard_set = updated_jaccard_dict[cur_filename]
        remained_related = []
        remained_rel_col = []
        for item in updated_jaccard_dict[cur_filename]:
            if not item in current_dict['related_table']:
                continue
            else:
                i = current_dict['related_table'].index(item)
                remained_related.append(current_dict['related_table'][i])
                remained_rel_col.append(current_dict['related_cols'][i])
        
        updated_dict = current_dict
        updated_dict['related_table'] = remained_related
        updated_dict['related_cols'] = remained_rel_col
        out_test.append(updated_dict)


        
for i in trange(len(out_test)):
    with jsonlines.open(outpath1, "a") as writer:
        writer.write(out_test[i])

for i in trange(len(out_train)):
    with jsonlines.open(outpath2, "a") as writer:
        writer.write(out_train[i])
