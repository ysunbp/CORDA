import jsonlines
import json
import csv
import os
from tqdm import tqdm
import numpy as np
import time
from tqdm import trange

import spacy
from collections import Counter
import re
import scipy
import torch
import torch.nn as nn
import random
from sklearn.model_selection import StratifiedKFold
import operator

#Generate Jaccard distance files

def jaccard(list1, list2): # compute the jaccard value between two lists
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1)+len(list2)) - intersection
    return float(intersection)/union

def read_tables(path): # load table content
    csv_reader = csv.reader(open(path))
    cur_set = []
    for i, line in enumerate(csv_reader):
        if i > 0:
            cur_set += line
    return list(set(cur_set))

def compute_jaccard(pathways, cur_path="../data/webtables/K4/"): # compute the jaccard distance between any two tables
    table_content = {}
    K4_content = {}
    jaccard_dict = {}
    for pathway in pathways:
        for file_name in os.listdir(pathway):
            file_path = pathway + file_name
            table_content[pathway[-3:]+file_name] = read_tables(file_path)
    for file_name in os.listdir(cur_path):
        file_path = cur_path+file_name
        K4_content[cur_path[-3:]+file_name] = read_tables(file_path)
    total_length = len(table_content)
    K4_length = len(K4_content)
    for i in trange(K4_length):
        list1_key = list(K4_content.keys())[i]
        if os.path.exists('./jaccard/'+list1_key[:-3]+'json'):
            continue
        list1 = K4_content[list1_key]

        for j in range(total_length):
            list2_key = list(table_content.keys())[j]
            if list2_key == list1_key:
                continue
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
        output_dict = {}
        output_dict['filename'] = list1_key
        output_dict['jaccard_tables'] = sorted(jaccard_dict[list1_key], key=lambda item: item[1], reverse=True)
        with open('./jaccard/'+list1_key[:-3]+'json', 'w') as f:
            json.dump(output_dict, f)


def read_csv(csv_dir): # process the raw tables in csv format
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


def generate_base_json():
    table_base = './raw_data/K'
    out_base = './jsons/K' #base json files
    for round in range(5):
        out_dir = out_base+str(round)+'/'
        table_dir = table_base+str(round)+'/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        i = 0
        for file_name in os.listdir(table_dir):
            table_file = table_dir+file_name
            json_file = out_dir+str(i)+'.json'
            header, content, num_col, num_row, num_cell = read_csv(table_file)
            for j in range(num_col):
                dict = {}
                dict['filename'] = file_name[:-4]
                dict['headers'] = header
                dict['content'] = content
                dict['target'] = j
                dict['label'] = header[j].lower()
                with open(json_file, 'w') as out_json:
                    json.dump(dict, out_json)
                i += 1

def read_tables(path): # load table content
    csv_reader = csv.reader(open(path))
    cur_set = []
    for i, line in enumerate(csv_reader):
        if i > 0:
            cur_set.append(line)
    return cur_set


def containenglish(string): # check if there are English words
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

def match_date(datestr): # divide DATE
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

def match_name(namestr): # divide PERSON
    if '.' in namestr:
        data_type = 'PERSON1'
    else:
        data_type = 'PERSON2'
    return data_type

def NER_extraction(table_dir, out_dir): # Extract the NEs for each table
    nlp = spacy.load('en_core_web_trf')
    rounds = [0,1,2,3,4]
    for round in rounds:
        csv_dir = table_dir+str(round)
        out_json_dir = out_dir+str(round)
        for csv_file in tqdm(os.listdir(csv_dir)):
            csv_file_path = csv_dir+'/'+csv_file
            out_json_file_path = out_json_dir+'/'+csv_file[:-3]+'json'
            table_content = np.array(read_tables(csv_file_path))
            width = len(table_content[0])
            col_ne = []
            most_common = []
            new_dict = {}
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



def preprocess_date_name(): # process the DATE and PERSON types, output washed json files
    json_summary_base = './jsons/K'
    out_base = './jsons/K'
    table_base = './raw_data/K'
    rounds = [0,1,2,3,4]
    print('Now process the dates, quantities and names format')
    for round in rounds:
        json_summary_dir = json_summary_base+str(round)
        out_dir = out_base+str(round)
        table_dir = table_base+str(round)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(json_summary_dir):
            os.makedirs(json_summary_dir)
        for json_file_name in tqdm(os.listdir(json_summary_dir)):
            json_summary_path = json_summary_dir+'/'+json_file_name
            out_path = out_dir+'/'+json_file_name
            table_path = table_dir+'/'+json_file_name[:-4]+'csv'
            with open(json_summary_path, 'r') as jf:
                content = json.load(jf)
            col_types = content['most_common']
            table_content = np.array(read_tables(table_path))
            out_json = content
            out_json['NER_counts'] = []
            for i, data_type in enumerate(col_types):
                if data_type == 'DATE':
                    candidate_str = table_content[0, i]
                    if not candidate_str:
                        candidate_str = table_content[1, i]
                    candidate_type = match_date(candidate_str)
                    out_json['most_common'][i] = candidate_type
                if data_type == 'PERSON':
                    candidate_str = table_content[0, i]
                    if not candidate_str:
                        candidate_str = table_content[1, i]
                    candidate_type = match_name(candidate_str)
                    out_json['most_common'][i] = candidate_type
                if data_type == 'QUANTITY':
                    candidate_str = table_content[0, i]
                    if not candidate_str:
                        candidate_str = table_content[1, i]
                    candidate_type = match_quantity(candidate_str)
                    out_json['most_common'][i] = candidate_type
                NER_list = content['table_NE'][i]
                NER_set = set(NER_list)
                NER_dict = {}
                for item in NER_set:
                    NER_dict.update({item:NER_list.count(item)})
                if len(NER_dict) == 0:
                    NER_dict['EMPTY'] = 1
                out_json['NER_counts'].append(NER_dict)
        
            with open(out_path, 'w') as of:
                json.dump(out_json, of)


def editDistance(str1, str2, m, n): # Compute the edit distance between two strings
 
    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n
 
    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m
 
    # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.
    if str1[m-1] == str2[n-1]:
        return editDistance(str1, str2, m-1, n-1)
 
    # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.
    return 1 + min(editDistance(str1, str2, m, n-1),    # Insert
                   editDistance(str1, str2, m-1, n),    # Remove
                   editDistance(str1, str2, m-1, n-1)    # Replace
                   )


def generate_distance(): # Compute the edit distances between any two table schemata and save as json files
    type_dict = {'CARDINAL':'A', 'DATE1':'B','DATE2':'C', 'DATE3':'D', 'DATE4':'E', 'DATE5':'F', 'EVENT':'G', 'FAC':'H', 'GPE':'I', 'LANGUAGE':'J', 'LAW':'K', 'LOC':'L', 'MONEY':'M', 'NORP':'N', 'ORDINAL':'O', 'ORG':'P', 'PERCENT':'Q', 'PERSON1':'R','PERSON2':'S', 'PRODUCT':'T', 'QUANTITY':'U', 'TIME':'V', 'WORK_OF_ART':'W', 'EMPTY':'X'}
    washed_json_base = './jsons/K'
    out_dir_base = './distance-files/K'
    rounds = [0,1,2,3,4]
    table_strings = {}

    for round in rounds:
        json_summary_dir = washed_json_base+str(round)
        if not os.path.exists(json_summary_dir):
            os.makedirs(json_summary_dir)
        for json_file_name in tqdm(os.listdir(json_summary_dir)):
            json_summary_path = json_summary_dir+'/'+json_file_name
            with open(json_summary_path, 'r') as jf:
                content = json.load(jf)
            col_types = content['most_common']
            table_name = str(round)+json_file_name
            table_string = ''
            for col in col_types:
                table_string += type_dict[col]
            table_strings[table_name] = table_string

    for table_2 in tqdm(table_strings):
        cur_table = table_2[1:]
        cur_round = table_2[0]
        distances = {}
        for table_1 in table_strings:
            if table_strings[table_1] == table_strings[table_2]:
                if not '0' in distances.keys():
                    distances['0']=[table_1]
                    distances['0-type']=[table_strings[table_1]]
                else:
                    distances['0'].append(table_1)
                    distances['0-type'].append(table_strings[table_1])
        if not os.path.exists(out_dir_base+str(cur_round)+'/'):
            os.makedirs(out_dir_base+str(cur_round)+'/')
        with open(out_dir_base+str(cur_round)+'/'+cur_table, 'w') as of:
            json.dump(distances, of)

##########Compute the edit distance files



##########Use Jaccard to filter the tables and generate input

def generate_json():
    rounds = [0,1,2,3,4]
    distance_base = './distance-files/K'
    jaccard_base = './jaccard/K'
    table_base = './raw_data/K'
    json_base = './jsons/K' #base json files
    out_base = './sato_jsons/K'
    for round in rounds:
        json_dir = json_base+str(round)+'/'
        distance_dir = distance_base+str(round)+'/'
        jaccard_dir = jaccard_base+str(round)+'/'
        out_dir = out_base+str(round)+'/'
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        if not os.path.exists(distance_dir):
            os.makedirs(distance_dir)
        if not os.path.exists(jaccard_dir):
            os.makedirs(jaccard_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for json_file in tqdm(os.listdir(json_dir)):
            cur_file_path = json_dir+json_file
            with open(cur_file_path, 'r') as jsondict_file:
                content_dict = json.load(jsondict_file)
            cur_file_name = content_dict['filename']
            cur_target = content_dict['target']
            cur_label = content_dict['label']
                    
            try:
                cur_col = list(np.array(content_dict['content'])[:,int(cur_target)])
            except IndexError:
                cur_col = []
                for i, row in enumerate(content_dict['content']):
                    if i== 0:
                        general_width = len(row)
                    if not general_width == len(row):
                        continue
                    else:
                        cur_col.append(row[int(cur_target)])
                cur_col = list(cur_col)

            out_json = {}
            out_json['filename'] = cur_file_name
            out_json['target'] = cur_target
            out_json['label'] = cur_label
            out_json['col'] = cur_col
        

            cur_distance_path = distance_dir+cur_file_name+'.json'
            cur_file_jaccard = jaccard_dir+cur_file_name+'.json'
            with open(cur_distance_path, 'r') as relate:
                related = json.load(relate)
            with open(cur_file_jaccard, 'r') as jaccard:
                jaccard_dict = json.load(jaccard)

            related_candidate = related['0']
            related_type = related['0-type'][0]
            related_tables = []
            related_cols = []
            for item in jaccard_dict['jaccard_tables']:
                cur_round = item[0][1]
                cur_file = (item[0][1]+item[0][3:])[:-3]+'json'
                cur_score = item[1]
                if cur_score < 0.1:
                    continue
                else:
                    if cur_file in related_candidate:
                        if os.path.exists(table_base+cur_round+'/'+cur_file[1:-4]+'csv'):
                            related_tables.append(cur_file)
                            list_file = []
                            with open(table_base+cur_round+'/'+cur_file[1:-4]+'csv', 'r') as csv_file:
                                all_lines=csv.reader(csv_file)  
                                for one_line in all_lines:  
                                    list_file.append(one_line)  
                            list_file.remove(list_file[0])
                            arr_file = np.array(list_file)
                            related_cols.append(list(arr_file[:, int(cur_target)]))
                    
            out_json['related_tables'] = related_tables
            out_json['related_cols'] = related_cols
            with open(out_dir+json_file, 'w') as output:
                json.dump(out_json, output)
            

if __name__ == '__main__':

    pathways = ["./raw_data/K0/", "./raw_data/K1/", "./raw_data/K2/", "./raw_data/K3/", "./raw_data/K4/"]
    for round in range(5):
        compute_jaccard(pathways, cur_path="./raw_data/K"+str(round)+"/")

    table_dir = './raw_data/K'
    out_dir = './jsons/K'
    generate_base_json()
    NER_extraction(table_dir, out_dir) #extract the named entities from the webtables dataset
    preprocess_date_name()
    generate_distance()
    generate_json()

    json_base_dir = './sato_jsons/K'
    rounds = [0,1,2,3,4]
    keys = []
    for round in rounds:
        json_base_path = json_base_dir+str(round)+'/'
        with jsonlines.open("./rels"+str(round)+".jsonlines", 'w') as w:
            for filename in tqdm(os.listdir(json_base_path)):
                json_file_path = json_base_path+filename
                with open(json_file_path, 'r') as cur_file:
                    data_dict = json.load(cur_file)
                    cur_file_name = data_dict['filename']
                    if cur_file_name in keys:
                        continue
                    else:
                        keys.append(cur_file_name)
                    cur_rels = []
                    for item in data_dict['related_tables']:
                        out_item = item[1:-5]
                        cur_rels.append(out_item)
                    w.write({cur_file_name: cur_rels})