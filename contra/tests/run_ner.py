import sys

sys.path.append('/home/shunita/fairemb/')
import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from descs_and_models import DESCS_AND_MODELS
import torch
from contra.constants import SAVE_PATH

label_list = ['O', 'B', 'I']
label_encoding_dict = {'O': 0, 'B': 1, 'I': 2}
# label_encoding_dict = {'I-PRG': 2, 'I-I-MISC': 2, 'I-OR': 6, 'O': 0, 'I-': 0, 'VMISC': 0, 'B-PER': 3, 'I-PER': 4,
#                        'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8, 'B-MISC': 1, 'I-MISC': 2}

task = 'ner'
# dataset_name = 'NCBI-disease'
dataset_name = 'BC5CDR-disease'
# model_checkpoint = "distilbert-base-uncased"
# tokenizer_name = 'google/bert_uncased_L-2_H-128_A-2'
tokenizer_name = 'bert-base-uncased'
# RUN_MODEL_INDEX = 3
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def get_all_tokens_and_ner_tags(directory):
    return pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in
                      os.listdir(directory)]).reset_index().drop('index', axis=1)


def get_tokens_and_ner_tags(filename):
    # Works for NCBI files! yay!
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})


def get_token_dataset(dataset_directory):
    train_df = get_tokens_and_ner_tags(os.path.join(dataset_directory, 'train.tsv'))
    val_df = get_tokens_and_ner_tags(os.path.join(dataset_directory, 'train_dev.tsv'))
    test_df = get_tokens_and_ner_tags(os.path.join(dataset_directory, 'test.tsv'))

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (train_dataset, val_dataset, test_dataset)


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    metric = load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


def run_flow():
    train_dataset, val_dataset, test_dataset = get_token_dataset(f'data/NER/{dataset_name}')
    # after this map, the features will be: 'attention_mask', 'input_ids', 'labels', 'ner_tags', 'token_type_ids', 'tokens'
    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

    # model_name, model_checkpoint = DESCS_AND_MODELS_READMISSION[RUN_MODEL_INDEX]
    # model_name = 'bert_DERT_bertbase_new0.3_ref0.3_concat_anchor40_epoch19'
    model_name = 'bert_DERT_bertbase_new0.2_ref0.4_concat_anchor40_epoch19'
    # model_name = 'bert_base_uncased_2010_2018_v2020_epoch39'
    # model_name = 'bert_base_uncased_2010_2018_v2020_epoch19_take1'
    model_checkpoint = os.path.join(SAVE_PATH, model_name)
    # model_name = 'bert-base-uncased'
    # model_checkpoint = 'bert-base-uncased'


    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    args = TrainingArguments(
        f"{dataset_name}-{task}-{model_name}",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=1e-5, #0
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=test_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    # trainer.save_model(os.path.join(SAVE_PATH, f'ner_ncbi_{model_name}.model'))


if __name__ == '__main__':
    run_flow()