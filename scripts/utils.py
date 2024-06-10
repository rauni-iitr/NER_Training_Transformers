import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import torch
import spacy
import evaluate
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm

from transformers import (AutoTokenizer,
                          AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          Trainer, 
                          TrainingArguments)

from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from IPython.display import Markdown


import warnings
warnings.filterwarnings('ignore')

ner_labels = ['chronic_disease', 'cancer', 'treatment', 'allergy_name']
#ner lablels converted in iob format for data transformation into tokens and IOB entity list
ner_label_iob = ['O', 'B-CHR', 'I-CHR', 'B-CAN', 'I-CAN', 'B-TRE', 'I-TRE', 'B-ALL', 'I-ALL']

model_ckpt_in = ['bert-base-uncased', 'raunak6898/bert-finetuned-ner-t1',
                  'raunak6898/bert-finetuned-ner-t2','bert-base-cased']
model_ckpt_eval = ['raunak6898/bert-finetuned-ner-t1',
                  'raunak6898/bert-finetuned-ner-t2','raunak6898/bert-finetuned-ner-t3', 'raunak6898/bert-finetuned-ner-all_data']

id2label = {k: v for k, v in enumerate(ner_label_iob)}
label2id = {v: k for k, v in id2label.items()}

files_dir_path = '../data_source'

def idxs_to_remove(tags_list, text_list):
    idx_to_remove = []

    for i, x in enumerate(tags_list):
        mx_idx = -1
        for y in x:
            start_idx = int(y.split(':')[0])
            end_idx = int(y.split(':')[1])
            mx_idx = max(mx_idx, start_idx, end_idx)

            if(start_idx>end_idx):
                # print(y.split(':'))
                # print(i,x,"====", start_idx, end_idx)
                idx_to_remove.append(i)
            
            if(mx_idx > len(text_list[i])+1):
                # print(i, x, text_list[i])
                idx_to_remove.append(i)

    return idx_to_remove

def quick_eda(df):
    # print(df.info(), "\n")
    print("No. of duplicates in dataframe - {}".format(df.loc[df.duplicated()].shape[0]))
    
    tags = df['tags'].tolist()
    text = df['text'].tolist()
    tags = [[y for y in x.split(',') if len(y)!=0] for x in tags]
    set_labels = set([y.split(':')[-1] for x in tags for y in x])
    print("Unique labels in tags columns: {}".format(set_labels),"\n\n")

    return 

def create_entites(tags, text):
    entities = []
    for x in tags:
        temp = []
        for y in x:
            temp_ls = y.split(':')
            start_idx = int(temp_ls[0])
            end_idx = int(temp_ls[1])
            label = temp_ls[-1]
            tup = (start_idx, end_idx, label)
            # print(tup)
            temp.append(tup)
        entities.append(temp)
    
    return entities

def preprocess(df):
    print('cleaning .....')
    # print('original_shape: {}'.format(df.shape))
    df = df.drop_duplicates()
    # print('shape after duplicates drop: {}'.format(df.shape))
    tags = df['tags'].tolist()
    text = df['text'].tolist()
    tags = [[y for y in x.split(',') if len(y)!=0] for x in tags]
    # print(tags[:5])
    entities = create_entites(tags=tags, text=text)
    df['entities'] = entities
    remove_idx_ls = idxs_to_remove(tags, text)
    # print("{} indices to remove".format(len(remove_idx_ls)))
    # print('Removing_index: {}'.format(remove_idx_ls))
    df = df.drop(remove_idx_ls, axis=0).reset_index()

    # print("Final columns: {}".format(df.columns))
    return df


nlp = spacy.load('en_core_web_sm')

def remove_special_characters(s):
    """
    function to remove special characters which shall be used in get_token_iob_label_list,
        to remove special characters from initial tokens, 
        so that bracketed/abbreviated disease names etc can be include with the disease label
    """
    
    pattern = r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$'
    result = re.sub(pattern, '', s)
    return result

def get_token_iob_label_list(text_list, entities_list):
    """
    function to convert [text and entites] into [token_list and IOB entities] list respectively.
    text is converted to [spacy tokens list] 
    and the ent_text for the entites are put in a dictionary with ner_labels as keys.
    Every elements from [spacy token list] is checked if they are in the text of ner_label in ent_text dictionary
    and then using `beginning and inside flag` IOB entites are appended in [label_list]
    """
    
    tokens_list = []
    labels_list = []

    for text, entities in zip(text_list, entities_list):
        ents_text = {ele : "" for ele in ner_labels}

        for y in entities:
            # print(text[y[0]])
            # print(text[y[0]-1:y[1]])
            ents_text[y[-1]] += text[y[0]-1:y[1]]  ## start index always has one positive offset in data, hence y[0]-1

        ents_text[y[-1]] = [remove_special_characters(item) for item in ents_text[y[-1]].split()]
        # print(ents_text)
        doc = nlp(text)
        token_list = [token.text for token in doc]
        label_list = ['O']*len(token_list)

        cd_flag, c_flag, t_flag, a_flag = 0, 0, 0, 0
        for i, token in enumerate(token_list):
            if(token in ents_text['chronic_disease']):
                if(cd_flag==0):
                    label_list[i] = 'B-CHR'
                    cd_flag = 1
                    c_flag, t_flag, a_flag = 0, 0, 0
                else:
                    label_list[i] = 'I-CHR'
            
            elif(token in ents_text['cancer']):
                if(c_flag==0):
                    label_list[i] = 'B-CAN'
                    c_flag = 1
                    cd_flag, t_flag, a_flag = 0, 0, 0
                else:
                    label_list[i] = 'I-CAN'
                
            elif(token in ents_text['treatment']):
                if(t_flag==0):
                    label_list[i] = 'B-TRE'
                    t_flag = 1
                    cd_flag, c_flag, a_flag = 0, 0, 0
                else:
                    label_list[i] = 'I-TRE'
                
            elif(token in ents_text['allergy_name']):
                if(a_flag==0):
                    label_list[i] = 'B-ALL'
                    a_flag = 1
                    cd_flag, c_flag, t_flag= 0, 0, 0
                else:
                    label_list[i] = 'I-ALL'
            else:
                cd_flag, c_flag, t_flag, a_flag = 0, 0, 0, 0
                
        assert(len(token_list)==len(label_list)), "len of token_list and label_list mismatch at some iteration ."
        tokens_list.append(token_list)
        labels_list.append([label2id[label] for label in label_list])

    return (tokens_list, labels_list)
    

def align_labels_with_tokens(word_ids, labels):
    new_labels = []
    last_word = None
    for word_id in word_ids:
        if(word_id is None):
            new_labels.append(-100)
            last_word = None
        else:
            if(word_id!=last_word):
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
                last_word=word_id
            else:
                label = labels[word_id]
                if(label % 2 == 1):
                    label += 1
                new_labels.append(label)
    
    return new_labels

def tokenize_and_align(examples, tokenizer):
    """
    tokenizer function that shall be used to map datasetdicts into tokenized data with tensors, truncated and padded;
    and well aligned input_labels and labels, before training
    """
    
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    all_labels = examples['ner_tags']
    new_labels = []
    for i, label in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        labels = align_labels_with_tokens(word_ids=word_ids, labels=label)
        new_labels.append(labels)
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs



def final_process(df):
    try:
            df = df.drop(columns=['Unnamed: 0'])
    except:
        pass
    df = df.dropna(axis=0)
    quick_eda(df)
    df = preprocess(df)
    tokens, labels = get_token_iob_label_list(df['text'].tolist(), df['entities'].tolist())

    # print(x.shape, len(tokens), len(labels))
    df['tokens'] = tokens
    df['ner_tags'] = labels
    df = df[['tokens', 'ner_tags']]
    return df
    
def load_dataframes(dir_path:str, new_df: pd.DataFrame = None):
    dfs = []
    for f in glob(dir_path+'/*.xlsx'):
        df = pd.read_excel(f)
        df = final_process(df)
        dfs.append(df)
    if(new_df is not None):
        dfs.append(final_process(new_df))
    return dfs

def get_datadicts(dfs):
    ls_datadicts = []
    for i, x in enumerate(dfs):
        temp_datadict = DatasetDict({'train' : Dataset.from_pandas(x)})
        # print(temp_datadict)
        if(i-1>=0):
            sample_prev_100 = ls_datadicts[i-1]['train'].select(np.random.randint(0, (ls_datadicts[i-1]['train'].num_rows), size=100))
            temp_datadict['train'] = concatenate_datasets([temp_datadict['train'], sample_prev_100])

        temp_datadict = temp_datadict['train'].train_test_split(0.2, seed=42, shuffle=True)
        val_data = temp_datadict['train'].train_test_split(0.1, shuffle=True)
        temp_datadict['validation'] = val_data['test']
        temp_datadict['train']= val_data['train']
        temp_datadict['test_prevs'] = temp_datadict['test']
        
        if(i-1>=0):
            temp_datadict['test_prevs'] = concatenate_datasets([temp_datadict['test_prevs'], ls_datadicts[i-1]['test_prevs']])

        ls_datadicts.append(temp_datadict)
        # print(ls_datadicts)

    final_dataset = Dataset.from_pandas(pd.concat(dfs)).train_test_split(test_size=0.2, seed=42, shuffle=True)
    try:
        final_dataset = final_dataset.remove_columns(column_names=['__index_level_0__'])
    except:
        pass

    val_data = final_dataset['train'].train_test_split(0.1, shuffle=True)
    final_dataset['validation'] = val_data['test']
    final_dataset['train']= val_data['train']
    final_dataset['test_prevs'] = ls_datadicts[-1]['test_prevs']
    ls_datadicts.append(final_dataset)
    # print(len(ls_datadicts))
    for i in range(len(ls_datadicts)):
        if(i!=len(ls_datadicts)-2):
            ls_datadicts[i] = DatasetDict({'test_prevs' : ls_datadicts[i]['test_prevs']})
        else:
            continue
    dfs = []
    return ls_datadicts

def measures_out(results_self_ls):
    df = pd.DataFrame()
    for i, dict in enumerate(results_self_ls):
        # print(dict)
        temp = {k.split('_')[-1]: v for k, v in dict.items() if k[0]=='o'}
        temp['number'] = temp.pop('accuracy')
        temp['number'] = 0
        # print(temp)
        one = {k :v for k, v in dict.items() if k[0]!='o'}
        one['overall_weighted_f1'] = temp
        # print(one)
        df = pd.concat([df, pd.DataFrame(one)])

    index = [x for x in df.index.tolist() if x=='f1']
    df = df.loc[index].drop_duplicates()
    ['chronic_disease', 'cancer', 'treatment', 'allergy_name']
    columns = {'ALL': 'allergy_name',
            'CAN': 'cancer',
            'TRE': 'treatment',
            'CHR': 'chronic_disease'}
    df = df.rename(columns=columns)
    df.index = ['T1+T2+T3']
    return df.transpose()


# print(len(ls_datadicts))