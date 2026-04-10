import os
import gc
import json
import random
import argparse
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
import copy

class ARDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip()) 

                full_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": data['prompt']}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                full_text += data['response']
                encoded = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                )

                prompt_len = len(
                    tokenizer(
                        tokenizer.apply_chat_template([{"role": "user", "content": data['prompt']}], tokenize=False, add_generation_prompt=True),
                        add_special_tokens=True
                    )['input_ids']
                )

                self.samples.append(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "prompt_length": prompt_len,
                        "is_safe": data['is_safe'] if 'is_safe' in data else True
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    # we can also add dataset to dataset
    def add_dataset(self, new_dataset, num_samples=None):
        if isinstance(new_dataset, ARDataset):
            if num_samples is not None: self.samples.extend(random.sample(new_dataset.samples, num_samples))
            else: self.samples.extend(new_dataset.samples)
        else:
            raise ValueError("new_dataset must be a Dataset or DatasetDict")

def collate_fn(batch):
    EOS_TOKEN_ID = 126081
    input_ids_list = [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch]
    prompt_lengths = torch.tensor(
        [sample["prompt_length"] for sample in batch], dtype=torch.long
    )
    eot_positions = [len(sample["input_ids"]) - 1 for sample in batch]  # EOS token position
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=EOS_TOKEN_ID
    )

    is_safe = torch.tensor([sample["is_safe"] for sample in batch], dtype=torch.bool)

    # print(input_ids)
    return {"input_ids": input_ids, "prompt_lengths": prompt_lengths, "eot_positions": eot_positions, "is_safe": is_safe}

def forward_process(input_ids, eps=1e-3):
    MASK_ID = 126336
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(
        masked_indices, torch.full_like(input_ids, MASK_ID), input_ids
    )
    return noisy_batch, masked_indices, p_mask

def train(args):
    print("SAFEMASK Training Started...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.model_type in ["llada_1.5", "llada_instruct"]:
        base_model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device)
    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj", "ff_out"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config).to(device)
    model.train()

    for name, param in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.zero_()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    dataset = ARDataset(args.data_path, tokenizer)
    # retain_math = ARDataset("GSM8K.jsonl", tokenizer)
    # dataset.add_dataset(retain_math, num_samples=3000)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")
        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for step, batch in enumerate(tqdm(dataloader)):
            input_ids, prompt_length, is_safe = batch["input_ids"].to(device), batch["prompt_lengths"], batch["is_safe"].to(device)
            eot_positions = batch["eot_positions"]

            noisy_batch, masked_indices, p_mask = forward_process(input_ids)

            # Make sure the prompt tokens are not masked
            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_length.unsqueeze(1).to(noisy_batch.device))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

            # Calculate the answer length (including the padded <EOS> tokens)
            prompt_mask = prompt_mask.to(torch.int64)    
            answer_length = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
            answer_length = answer_length.repeat(1, noisy_batch.shape[1])

            input_ids[~is_safe] = 126081

            masked_indices = (noisy_batch == 126336)

            logits = model(input_ids=noisy_batch).logits

            token_loss = F.cross_entropy(
                logits[masked_indices],
                input_ids[masked_indices],
                reduction='none'
            ) / p_mask[masked_indices]

            loss = torch.sum(token_loss / answer_length[masked_indices]) / input_ids.shape[0]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_sum += loss.item()
            epoch_step_count += 1

            torch.cuda.empty_cache()
            gc.collect()

            print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

        avg_loss = epoch_loss_sum / epoch_step_count
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        if (epoch+1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_dir = f"model/{args.model_type}_{args.method}/lora_epoch{epoch+1}"
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved model to {save_dir}")

    print("Training complete!")
