import os
import torch
import numpy as np
import argparse
import random
from method import train

def fix_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mode", choices=["data"], default="data")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--method', type=str, default="a2d")
    parser.add_argument("--data_path", type=str, default="benchmarks/beavertails.jsonl")
    parser.add_argument("--model_type", type=str, default="llada_instruct", choices=["llada_1.5", "llada_instruct", "dream_instruct"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()

def model_name_from_model_type(model_type):
    if model_type == "llada_instruct": return "GSAI-ML/LLaDA-8B-Instruct"
    elif model_type == "llada_1.5": return "GSAI-ML/LLaDA-1.5"
    elif model_type == "dream_instruct": return "Dream-org/Dream-v0-Instruct-7B"
    else: raise ValueError(f"Model type {model_type} not supported")

def main(args):

    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Mode: {args.mode}")
    print(f"Save Interval: {args.save_interval}")
    print(f"Data Path: {args.data_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Seed: {args.seed}")
    train(args)

if __name__ == "__main__":
    args = parse_args()
    args.model_path = model_name_from_model_type(args.model_type)
    fix_random_seed(args.seed)

    main(args)