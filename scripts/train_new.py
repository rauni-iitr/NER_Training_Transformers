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

from utils import *

import warnings
warnings.filterwarnings('ignore')

# print(id2label)

last_model_ckpt = model_ckpt_eval[-2]

def create_datadict(data: pd.DataFrame = None):
    dfs = load_dataframes(dir_path=files_dir_path, new_df=data)
    ls_datadicts = get_datadicts(dfs)
    # print(ls_datadicts)
    return ls_datadicts

new_df = pd.read_excel('../data_source/G3.xlsx')

# create_datadict(new_df)
metric = evaluate.load('seqeval')

def compute_metrics(eval_preds):
    """Compute metrics for training pipeline to evaluate metrics on validations set while training"""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens like [CLS] and [SEP] or any unknown tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def train_new(data: pd.DataFrame, model_ckpt_out:str = 'new_fine_tine_ner'):
    """### takes last model trained chekpoint in continual fashion and trains a model and saves at an output_model_ckpt,
      check arguments when loading from disk """

    ls_datadicts = create_datadict(data=data)
    print('all_data_collected_for train')

    model_ckpt = last_model_ckpt
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_ckpt)
    data_collater = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(model_ckpt, id2label=id2label, label2id=label2id)

    tokenized_datasets = ls_datadicts[-2].map(
        lambda batch: tokenize_and_align(batch, tokenizer=tokenizer),
        batched=True,
        remove_columns=ls_datadicts[-2]["train"].column_names
    )

    args = TrainingArguments(
        output_dir=model_ckpt_out,
        evaluation_strategy='epoch',
        # logging_steps=100,
        # logging_strategy='steps',
        save_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        use_mps_device=True,
        load_best_model_at_end=True,
        # push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collater,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    return trainer.save_model(output_dir=model_ckpt_out)

def evaluate_model(model_ckpt:str):
    """ use to evaluate any model -  genertaes a csv metric table output"""

    ls_datadicts = create_datadict()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_ckpt)
    data_collater = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(model_ckpt, id2label=id2label, label2id=label2id)
    trainer = Trainer(model=model, data_collator=data_collater, tokenizer=tokenizer)

    # tokenized_datasets = ls_datadicts[:3].map(
    #     lambda batch: tokenize_and_align(batch, tokenizer=tokenizer),
    #     batched=True,
    #     remove_columns=ls_datadicts[-2]["test_prevs"].column_names
    # )
    df_g_all = pd.read_csv('../scripts/submission_f1.csv')
    df_g_all.index = df_g_all.iloc[:,0].values
    df_g_all.drop(columns=['Unnamed: 0'], inplace=True)
    df_g_all = df_g_all[['G1+G2+G3']]

    df_out = pd.DataFrame()

    for iter in range(3):
        tokenized_dataset = ls_datadicts[iter].map(
            lambda batch: tokenize_and_align(batch, tokenizer=tokenizer),
            batched=True,
            remove_columns=ls_datadicts[iter]["test_prevs"].column_names
        )

        out = trainer.predict(tokenized_dataset['test_prevs'])

        metric = evaluate.load('seqeval')
        for i in range(len(tokenized_dataset['test_prevs'])):
            y_true = [id2label[id] for id in tokenized_dataset['test_prevs'][i]['labels'] if id != -100]
            y_pred = [id2label[id] for id in np.argmax(out[0][i], axis=1)[1:len(y_true)+1]]
            metric.add_batch(predictions=[y_pred], references=[y_true])

        results_self_prevs = [metric.compute()]
        b = measures_out(results_self_prevs)
        df_out = pd.concat([df_out, b], axis=1)
    cols = ['T1', 'T1+T2','T1+T2+T3']
    df_out.columns = cols
    df_out = pd.concat([df_out, df_g_all], axis=1)
    return df_out.to_csv('./metrics_out.csv')


# train_new(new_df)

evaluate_model(model_ckpt='raunak6898/bert-finetuned-ner-t3')

        



    




    