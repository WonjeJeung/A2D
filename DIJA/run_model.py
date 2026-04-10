import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import numpy as np
import pdb
import random 

model_id = "GSAI-ML/LLaDA-8B-Instruct"

def prepare_prompt(tokenizer, behavior: str, is_instruct: bool) -> str:
    """Prepare prompt based on model type (Instruct or base)"""
    if is_instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": behavior}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return behavior

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_tokenized_input(prompt: str, tokenizer, device: str) -> dict:
    """Tokenize prompt and move to device"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask

def generate_response(
    prompt: str,
    tokenizer,
    model,
    args
) -> str:
    """Generate response using diffusion-style generation"""

    input_ids, attention_mask = get_tokenized_input(prompt, tokenizer, model.device)
    print(args)

    # Generate using diffusion method
    output_ids = generate_llada(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        gen_length=args['gen_length'],
        temperature=0.2,
        mask_id=args['mask_id']
    )

    response = tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]

    return response

def generate_llada(
    input_ids,
    attention_mask,
    model,
    gen_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x = input_ids
        # feature_cache = dLLMCache()
        # feature_cache.reset_cache(0) 
        num_transfer_tokens = 1
        counter = 0
        while (x == mask_id).any():
            mask_index = x == mask_id
            # pdb.set_trace()
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == "entropy":

            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(
                mask_index, x0, x
            )
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens
                    ).indices
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            # pdb.set_trace()
            counter += 1
            if counter > 57:
                break
        # Ensure the output is of the correct length
        return x

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to('cuda').eval()

    prompt = "Hello World"
    prompt = prepare_prompt(tokenizer, prompt, is_instruct=True)

    print("Prompt prepared for the model:", prompt)
    MASK_TOKEN = "<|mdm_mask|>"  # Mask token for diffusion
    mask_counts = 128  # Number of mask tokens to add
    END_TOKEN = "<endoftext>"

    print("Mask token ID:", MASK_TOKEN, "End token ID:", END_TOKEN)
    prompt += MASK_TOKEN * mask_counts + END_TOKEN

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        args={
            'steps': 128,
            'gen_length': 128,
            'mask_id': 126336
        }
    )

    print("Generated response:", response)

main()