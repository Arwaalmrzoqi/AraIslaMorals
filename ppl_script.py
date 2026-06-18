import argparse
import sys
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset
import torch
from tqdm import tqdm
import pickle
import random
import os
import difflib
import numpy as np
import torch
import json

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def concat(example):
    example["prompt"] = example["base_norm"] + " " + example["descriptive_norm"] + " " + example["situation"] + " " + example["intention"]
    return example

def load_data(data_file):
    print('Download dataset...')
    if data_file.endswith(".json"):
        dataset = load_dataset('json', data_files=data_file, split='train')
    elif data_file.endswith('.csv'):
        dataset = load_dataset('csv', data_files=data_file, split='train')
    dataset = dataset.map(concat)
    dataset = dataset.rename_column("moral_action", "chosen")
    dataset = dataset.rename_column("immoral_action", "rejected")

    dataset = dataset.remove_columns(
        ['base_norm', "descriptive_norm", 'situation', 'intention', 'moral_consequence', 'immoral_consequence']
    ) 
    return dataset

def load_model(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device


def compute_perplexity(model_name, model, tokenizer, dataset, device, dataset_name, output_path=None):
    count_moral = 0
    ppl_moral, ppl_immoral = [], []

    for dat in tqdm(dataset):
        input_all = tokenizer(dat["prompt"], return_tensors="pt")

        input = tokenizer(dat["chosen"], return_tensors="pt")
        input["labels"] = torch.hstack([torch.full_like(input_all["input_ids"], -100), input["input_ids"]])
        input["input_ids"] = torch.hstack([input_all["input_ids"], input["input_ids"]])
        input["attention_mask"] = torch.hstack([input_all["attention_mask"], input["attention_mask"]])
        input.to(device)
        output = model(**input)
        loss_chosen = output.loss.item()
        ppl_moral.append(loss_chosen)

        input = tokenizer(dat["rejected"], return_tensors="pt")
        input["labels"] = torch.hstack([torch.full_like(input_all["input_ids"], -100), input["input_ids"]])
        input["input_ids"] = torch.hstack([input_all["input_ids"], input["input_ids"]])
        input["attention_mask"] = torch.hstack([input_all["attention_mask"], input["attention_mask"]])
        input.to(device)
        output = model(**input)
        loss_rejected = output.loss.item()
        ppl_immoral.append(loss_rejected)

        if loss_chosen < loss_rejected:
            count_moral += 1

    count_immoral_preferred, count_moral_preferred = 0, 0

    for a, b in zip(ppl_moral, ppl_immoral):
        if a > b:
            count_immoral_preferred += 1
        elif b > a:
            count_moral_preferred += 1

    print("Input:", dataset_name)
    print("Model:", model_name)
    print("=" * 100)
    print('Count moral preferred | immoral preferred :', count_moral_preferred, ":", count_immoral_preferred)
    print('Average perplexity moral:', round(torch.mean(torch.tensor(ppl_moral)).item(), 2), "~",
          round(torch.std(torch.tensor(ppl_moral)).item(), 2))
    print('Average perplexity immoral:', round(torch.mean(torch.tensor(ppl_immoral)).item(), 2), "~",
          round(torch.std(torch.tensor(ppl_immoral)).item(), 2))
    print('Percentage moral preferred', count_moral / len(dataset))
    print("=" * 100)

    result = {'dataset': dataset_name, 'model': model_name,
              'count_moral': count_moral_preferred,
              'count_immoral': count_immoral_preferred,
              'avg_ppl_moral': round(torch.mean(torch.tensor(ppl_moral)).item(), 2),
              'std_ppl_moral': round(torch.std(torch.tensor(ppl_moral)).item(), 2),
              'avg_ppl_immoral': round(torch.mean(torch.tensor(ppl_immoral)).item(), 2),
              'std_ppl_immoral': round(torch.std(torch.tensor(ppl_immoral)).item(), 2),
              'prct_moral_preferred': round(count_moral / len(dataset) * 100, 2)
              }

    if output_path:
        result_file = output_path
        os.makedirs(os.path.dirname(result_file) or '.', exist_ok=True)
    else:
        result_file = 'result_ppl.json'

    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
    print(f"Results saved to: {result_file}")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument parser for training script.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-1.7B", help='Model name')
    parser.add_argument('--ref_model', type=str, default='Qwen', help='Reference model')
    parser.add_argument('--data_file', type=str, default='dataset.json', help='json file that has moral and immoral examples')
    parser.add_argument('--output', type=str, default=None, help='Output path for the results JSON file')

    args = parser.parse_args()

    if args.hf_token is None:
        print('HuggingFace token not provided, please provide it using --hf_token')
        sys.exit(1)

    dataset = load_data(args.data_file)
    model, tokenizer, device = load_model(args.model_name, args.hf_token)
    compute_perplexity(args.model_name, model, tokenizer, dataset, device, args.data_file, args.output)