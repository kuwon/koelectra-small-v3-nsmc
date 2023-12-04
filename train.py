import os
import json
import argparse
import logging
import sys
import numpy as np
import torch
from datasets import load_from_disk, load_dataset, load_metric
from transformers import (
    ElectraModel, ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments, set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json


def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=False)    
    parser.add_argument("--tokenizer_id", type=str, default='monologg/koelectra-small-v3-discriminator')
    parser.add_argument("--model_id", type=str, default='monologg/koelectra-small-v3-discriminator')
    
    # For User Setting
    parser.add_argument("--ds_name", type=str, default='invst_opinion')

    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


# compute metrics function for binary classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    
    # Set up logging 
    logging.basicConfig(
        level=logging.INFO, 
        format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    args = parser_args()    
    n_gpus = '1' #torch.cuda.device_count() # for mps
    os.environ["GPU_NUM_DEVICES"] = n_gpus
    os.environ['CURL_CA_BUNDLE'] = '' # for ssl error

    chkpt_dir = 'chkpt'
    dataset_dir = f'datasets/{args.ds_name}'
    model_dir = 'model'
    output_data_dir = 'data'


    logger.info("***** Arguments *****")    
    logger.info(''.join(f'{k}={v}\n' for k, v in vars(args).items()))
    
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)    

    tokenizer_id = args.tokenizer_id
    model_id = args.model_id

    #tokenizer_id = 'daekeun-ml/koelectra-small-v3-nsmc'
    #model_id = "daekeun-ml/koelectra-small-v3-nsmc"

    # load datasets
    ds = load_from_disk(dataset_dir).class_encode_column('label').train_test_split(test_size=0.2)
    train_dataset = ds['train']
    test_dataset = ds['test']

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    logger.info(train_dataset[0])    
    # download tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_id)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['document'], padding='max_length', truncation=True)

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    cols = ['input_ids', 'attention_mask', 'label'] # for nsmc
    train_dataset.set_format('torch', columns=cols)
    test_dataset.set_format('torch', columns=cols)

    # Prepare model labels - useful in inference API
    labels = train_dataset.features["label"].names
    print(labels)

    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Set seed before initializing model
    set_seed(args.seed)
    
    # Download pytorch model
    model = ElectraForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    print(model.config)

    # define training args
    training_args = TrainingArguments(
        output_dir=chkpt_dir,
        overwrite_output_dir=True if get_last_checkpoint(chkpt_dir) is not None else False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        #fp16=args.fp16, # not working on mac m1
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        disable_tqdm=args.disable_tqdm,
        logging_dir=f"{output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics    
    )

    # train model
    if get_last_checkpoint(chkpt_dir) is not None:
        logger.info("***** Continue Training *****")
        last_checkpoint = get_last_checkpoint(chkpt_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
        
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(output_data_dir, "eval_results.txt"), "w") as writer:
        print("***** Evaluation results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            logger.info(f"{key} = {value}\n")

    # Saves the model
    trainer.save_model(model_dir)
    trainer.push_to_hub("solikang/koelectra-small-v3-nsmc")


if __name__ == "__main__":
    main()    
