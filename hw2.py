#!/usr/bin/env python

import os
import sys
import json
import argparse
import logging
import math
import numpy as np
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_norm_
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    default_data_collator,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    MT5ForConditionalGeneration, MT5Tokenizer,
    get_linear_schedule_with_warmup,
)


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Language Generation")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The name of the model.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--valid_file", type=str, default=None, help="A csv or a json file containing the validation data or testing data."
    )
    parser.add_argument(
        "--output_file", type=str, default="output.jsonl", help="A jsonl file intended to be output."
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help=(
            "max input length"
        ),
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=5,
        help=(
            "beams"
        ),
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=64,
        help=(
            "max output length"
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning Rate",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch Size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator()
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -%(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    logger.info(accelerator.state)

    model_name = args.model_name
    batch_size = args.batch
    lr = args.lr
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, do_lower_case=True)

    raw_datasets = load_dataset("json", data_files={"train": args.train_file, "valid": args.valid_file})
    column_names = raw_datasets["train"].column_names
    title_names = "title"
    text_names = "maintext"
    def prepare_train_features(examples, indices):
        inputs = examples[text_names]
        outputs = examples[title_names]
        all_inputs = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_input_length)
        with tokenizer.as_target_tokenizer():
            outputs = tokenizer(outputs, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_output_length)
        outputs["input_ids"] = [[(label if label != tokenizer.pad_token_id else -100) for label in labels] for labels in outputs["input_ids"]]
        all_inputs['labels'] = outputs['input_ids']
        all_inputs['indices'] = indices
        return all_inputs

    train_examples = raw_datasets["train"]
    train_dataset = train_examples.map(
        prepare_train_features,
        with_indices=True,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
     )
    print(train_dataset)
    def prepare_valid_features(examples):
        inputs = examples[text_names]
        outputs = examples[title_names]
        all_inputs = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_input_length)
        return all_inputs

    valid_examples = raw_datasets["valid"]
    valid_dataset  = valid_examples.map(
        prepare_valid_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
     )
    print(valid_dataset)
    print("Data preprocessing done!")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config).to('cuda')
    print("Model:",model)
    print("Model:", model.device)
    train_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=train_collator,
                                batch_size=batch_size, num_workers=4)
    valid_collator = default_data_collator
    valid_loader = DataLoader(valid_dataset, shuffle=False, collate_fn=valid_collator,
                                batch_size=batch_size, num_workers=4)


    num_epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_epochs*len(train_loader))

    progress_bar = tqdm(range(num_epochs*(int(len(train_loader)))), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Training loop
    print(">> Start training <<")
    for epoch in range(num_epochs):

        total_loss = 0

        for step, data in enumerate(train_loader):
            with accelerator.accumulate(model):
                model.train()
                optimizer.zero_grad()

                # Forward pass
                data = {key: value.to('cuda') for key, value in data.items() if key!="indices"}
                outputs = model(**data)

                loss = outputs.loss
                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                #print("loss ", loss.item())

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1


        # Print average loss for the epoch
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}")

    print("Save the model!")
    model.eval()
    gen_kwargs = {
            "max_length": args.max_output_length,
            "num_beams":  args.beams,
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.9,
            "temperature": 1.2,
            }
    predictions = []
    for step, data in enumerate(valid_loader):
        data = {key: value.to('cuda') for key, value in data.items()}
        with torch.no_grad():
            tokens = accelerator.unwrap_model(model).generate(
                        data["input_ids"],
                        attention_mask=data["attention_mask"],
                        **gen_kwargs
                    )
            tokens = accelerator.pad_across_processes(tokens, dim=1, pad_index=tokenizer.pad_token_id)
            tokens = accelerator.gather(tokens).cpu().numpy()
            pred = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            count = 0
            for pre in pred:
                pred[count] = pre.replace("<extra_id_0>", "")
                count = count+1
            predictions += pred
            print("Pred:", pred)
    print("Prediction: ", predictions)

    with open(args.output_file, 'w') as fout:
        for id_, pred in zip(valid_examples['id'], predictions):
            print(json.dumps({"id": id_, "title": pred}, ensure_ascii=False), file=fout)



if __name__ == "__main__":
    main()



"""
        # Eval Valid
        model.eval()
        gen_kwargs = {
                "max_length": args.max_output_length,
                "num_beams":  args.beams,
                "do_sample": False,
                }
        predictions = []
        for step, data in enumerate(valid_loader):
            data = {key: value.to('cuda') for key, value in data.items()}
            with torch.no_grad():
                tokens = accelerator.unwrap_model(model).generate(
                            data["input_ids"],
                            attention_mask=data["attention_mask"],
                            **gen_kwargs
                        )
                tokens = accelerator.pad_across_processes(tokens, dim=1, pad_index=tokenizer.pad_token_id)
                tokens = accelerator.gather(tokens).cpu().numpy()
                pred = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                count = 0
                for pre in pred:
                    pred[count] = pre.replace("<extra_id_0>", "")
                    count = count+1
                predictions += pred
                print("Pred:", pred)
        print("Prediction: ", predictions)
        with open(f"output_{epoch}.json", 'w') as fout:
            for id_, pred in zip(valid_examples['id'], predictions):
                print(json.dumps({"id": id_, "title": pred}, ensure_ascii=False), file=fout)
                """

