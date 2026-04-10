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

# ─── Fix Random Seed ──────────────────────────────────────────
def fix_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─── Argument Parser ──────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["base", "instruct"], default="instruct")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mode", choices=["double", "single", "multi"], default="double")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ─── Dataset Class ────────────────────────────────────────────
class ARDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())  # JSONL 읽기
                full_text = data["prompt"]
                encoded = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                )
                prompt_len = len(
                    tokenizer(
                        data["prompt"],
                        add_special_tokens=False
                    )["input_ids"]
                )
                self.samples.append(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "prompt_length": prompt_len,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    EOS_TOKEN_ID = 126081
    input_ids_list = [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch]
    prompt_lengths = torch.tensor(
        [sample["prompt_length"] for sample in batch], dtype=torch.long
    )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=EOS_TOKEN_ID
    )
    return {"input_ids": input_ids, "prompt_lengths": prompt_lengths}

# ─── Noising Function ─────────────────────────────────────────
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

# ─── Main Training ───────────────────────────────────────────
def main():
    args = parse_args()
    fix_random_seed(args.seed)  # 시드 고정

    model_map = {
        "instruct": "GSAI-ML/LLaDA-8B-Instruct",
        "base": "GSAI-ML/LLaDA-8B-Base"
    }

    data_map = {
        "double": "data/doublet_train.jsonl",
        "single": "data/singlet_train_1_modified.jsonl",
        "multi": "data/multilet_train.jsonl"
    }

    model_name = model_map[args.model_type]
    data_path = data_map[args.mode]
    run_name = f"LLaDA_train_{args.model_type}_{args.mode}_{args.learning_rate}"

    wandb.login(key="3408e88df50f40f9b1ca906187f2d85ca4b8aab8", relogin=True)
    wandb.init(
        project="templet syllogism train",
        name=run_name,
        entity="marcshin201-yonsei-university"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        model_name,
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

    dataset = ARDataset(data_path, tokenizer)
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

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)

            # Random length cropping
            if torch.rand(1).item() < 0.01:
                rand_len = torch.randint(
                    1, input_ids.shape[1] + 1, (1,)
                ).item()
                input_ids = input_ids[:, :rand_len]

            noisy_batch, masked_indices, p_mask = forward_process(input_ids)
            logits = model(input_ids=noisy_batch).logits

            token_loss = F.cross_entropy(
                logits[masked_indices],
                input_ids[masked_indices],
                reduction='none'
            ) / p_mask[masked_indices]

            loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_sum += loss.item()
            epoch_step_count += 1

            torch.cuda.empty_cache()
            gc.collect()

            wandb.log({"loss": loss.item()})
            if (step + 1) % 100 == 0:
                print(
                    f"[Epoch {epoch+1} Step {step+1}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss_sum / epoch_step_count
        wandb.log({"epoch_loss_avg": avg_loss, "epoch": epoch+1})
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

        if (epoch+1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_dir = f"model/LLaDA/{args.model_type}-{args.mode}-{args.learning_rate}/lora_epoch{epoch+1}"
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved model to {save_dir}")

    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()