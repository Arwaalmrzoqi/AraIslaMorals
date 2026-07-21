# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.26" peft accelerate bitsandbytes transformers
# !pip install -U git+https://github.com/huggingface/trl
import json
import sys
import pandas as pd
from trl import DPOConfig, DPOTrainer
import torch
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from tqdm import tqdm
import random
import os
import numpy as np
import argparse


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def concat(example):
    example["prompt"] = example["base_norm"] + " " + example["situation"] + " " + example["intention"]
    return example

def load_data(data_file, align_to_moral):
    print('Download dataset...')

    if data_file.endswith(".json"):
        dataset = load_dataset('json', data_files=data_file, split='train')
    elif data_file.endswith('.csv'):
        dataset = load_dataset('csv', data_files=data_file, split='train')

    dataset = dataset.map(concat)
    if align_to_moral:
        dataset = dataset.rename_column("moral_action", "chosen")
        dataset = dataset.rename_column("immoral_action", "rejected")
    else:
        dataset = dataset.rename_column("immoral_action", "chosen")
        dataset = dataset.rename_column("moral_action", "rejected")

    dataset = dataset.remove_columns(
        ['base_norm', 'descriptive_norm', 'situation', 'intention', 'moral_consequence', 'immoral_consequence'])
    dataset = dataset.train_test_split(test_size=0.3)
    return dataset

def qlora_training(model_name, nb_examples, seed, hf_token, dataset, epochs):
    max_seq_length = 2048

    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available. Training will run on CPU.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hf_token
    )

    if torch.cuda.is_available():
        model.to('cuda')
        print(f"Model moved to: {model.device}")
    else:
        print("Model is on CPU.")

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=seed,
    )

    training_args = DPOConfig(
        output_dir="./output",
        beta=0.1,
        fp16=False, 
        bf16=False,
        num_train_epochs = epochs,        
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"].shard(num_shards=int(8400 / nb_examples), index=0),
    )
    dpo_trainer.train()
    return dpo_trainer.model, tokenizer

def evaluate_model(model, tokenizer, dataset, model_name, align_to_moral, dataset_name, seed, output_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    count_moral = 0
    ppl_moral, ppl_immoral = [], []

    for dat in tqdm(dataset["test"]):
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

    print(count_moral / len(dataset["test"]))

    print("Model:", model_name)
    print("Dataset:", dataset_name)
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
        result_path = output_path
    elif align_to_moral:
        result_path = 'result_dpo_qlora_' + '_' + '_to_moral' + '.json'
    else:
        result_path = 'result_dpo_qlora_'  + '_to_immoral' + '.json'
    
    os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument parser for training script.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--nb_examples', type=int, default=8400, help='Number of training examples')
    parser.add_argument('--align_to_moral', choices=[True, False], default=True,
                        help='Configure DPO to prefer moral actions')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-1.7B", help='Model name')
    parser.add_argument('--ref_model', type=str, default='Qwen', help='Reference model')
    parser.add_argument('--data_file', type=str, help='json file that has the data')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--output', type=str, default=None, help='Output path for the results JSON file')
    args = parser.parse_args()

    if args.nb_examples > 8400:
        print('nb_examples must be lower than or equal to 8400')
        sys.exit(1)

    if args.hf_token is None:
        print('HuggingFace token not provided, please provide it using --hf_token')
        sys.exit(1)

    seed_everything(args.seed)

    dataset = load_data(args.data_file, args.align_to_moral)
    model, tokenizer = qlora_training(args.model_name, args.nb_examples, args.seed, args.hf_token, dataset, args.epochs)
    evaluate_model(model, tokenizer, dataset, args.model_name, args.align_to_moral, args.data_file, args.seed, args.output)
