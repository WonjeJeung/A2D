import os
import gc
import json
import random
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn

class DreamWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Dream base model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

class ARDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                # Chat template 적용
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
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": data['prompt']}],
                            tokenize=False,
                            add_generation_prompt=True
                        ),
                        add_special_tokens=True
                    )['input_ids']
                )

                self.samples.append({
                    "input_ids": encoded["input_ids"],
                    "prompt_length": prompt_len,
                    "is_safe": data['is_safe']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def forward_process_local(input_ids: torch.Tensor, mask_id: int, eps: float = 1e-3):
    device = input_ids.device
    b, l = input_ids.shape
    t = torch.rand(b, device=device)  # sentence-level noise
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].expand(b, l)
    masked_indices = torch.rand((b, l), device=device) < p_mask
    noisy_batch = torch.where(masked_indices, torch.full_like(input_ids, mask_id), input_ids)
    return noisy_batch, masked_indices, p_mask

def train(args):
    model_name = "Dream-org/Dream-v0-Instruct-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)

    base_model = DreamWrapper(base_model)

    # pad/mask token id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    mask_id = getattr(getattr(base_model, "config", object()), "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        try:
            mask_id = tokenizer.convert_tokens_to_ids("<mask>")
        except Exception:
            raise ValueError("mask_token_id를 찾을 수 없습니다. Dream 토크나이저의 mask 토큰을 확인하세요.")

    # LoRA
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

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # Dataset / Dataloader
    dataset = ARDataset(args.data_path, tokenizer)
    # dataset.samples = [sample for sample in dataset.samples if sample.get("is_safe", True)]

    def collate_fn(batch):
        input_ids_list = [torch.tensor(s["input_ids"], dtype=torch.long) for s in batch]
        prompt_lengths = torch.tensor([s["prompt_length"] for s in batch], dtype=torch.long)
        is_safe = torch.tensor([s["is_safe"] for s in batch], dtype=torch.bool)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        return {"input_ids": input_ids, "prompt_lengths": prompt_lengths, "is_safe": is_safe}

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Train loop
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")
        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            prompt_length = batch["prompt_lengths"]
            is_safe = batch["is_safe"].to(device)

            # forward_process
            noisy_batch, masked_indices, p_mask = forward_process_local(input_ids, mask_id=mask_id)

            # Prompt 토큰은 마스킹하지 않음
            token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
            prompt_mask = (token_positions < prompt_length.unsqueeze(1).to(noisy_batch.device))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

            # 안전하지 않은 샘플은 target을 EOS로 덮기
            if not is_safe.all():
                input_ids[~is_safe] = eos_id
                
            masked_indices = (noisy_batch == mask_id)

            # 모델 forward
            outputs = model(input_ids=noisy_batch)
            logits = outputs.logits  # [B, L, V]

            # one-token shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            label_mask = masked_indices[:, 1:]
            p_mask_shift = p_mask[:, 1:]

            if label_mask.any():
                sel_logits = shift_logits[label_mask]
                sel_labels = shift_labels[label_mask]
                sel_pmask = p_mask_shift[label_mask]
                token_loss = F.cross_entropy(sel_logits, sel_labels, reduction="none") / sel_pmask

                # 여기서 is_safe인 것에 대해서는 0.1배만 loss를 줌
                loss = token_loss.mean()
            else:
                loss = shift_logits.new_zeros(())

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_sum += float(loss.detach().cpu())
            epoch_step_count += 1

            torch.cuda.empty_cache()
            gc.collect()

        epoch_loss_avg = epoch_loss_sum / epoch_step_count if epoch_step_count > 0 else 0.0
        print(f"Epoch {epoch+1} Loss: {epoch_loss_avg:.4f}")

        if (epoch+1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_dir = f"model/{args.model_type}_{args.method}/lora_epoch{epoch+1}"
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved model to {save_dir}")

    print("Training complete!")