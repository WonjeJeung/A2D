import os
import re
import sys
import json
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch.nn as nn

import pandas as pd
import torch.nn.functional as F
import torch.distributions as dists
from typing import Optional, Union, Dict, Any, Tuple, List
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)
from defense_utils import Defender
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
from transformers import __version__
logger = logging.get_logger(__name__)


# Constants
DEFAULT_GEN_LENGTH = 64
DEFAULT_STEPS = 64
DEFAULT_MASK_ID = 151666
DEFAULT_MASK_COUNTS = 36
MASK_TOKEN = "<|mask|>"
SPECIAL_TOKEN_PATTERN = r'<mask:(\d+)>'
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        criteria=10,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            decoding=kwargs.get("decoding", None)
        )

        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        decoding,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        EOS_TOKEN_ID = generation_config.eos_token_id if isinstance(generation_config.eos_token_id, int) else generation_config.eos_token_id[0]

        # for i in range(steps):
        while (x == mask_token_id).any():
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            # logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            mean_of_eos = logits[mask_index][:1]
            mean_of_eos = F.softmax(mean_of_eos, dim=-1)
            # print("mean_of_eos", mean_of_eos[:, EOS_TOKEN_ID].mean().item())
            # if mean_of_eos[:, EOS_TOKEN_ID].mean() > criteria and i < 1:
            #     return DreamModelOutput(
            #         sequences="Sorry, I can't help you with that to follow the safety policy.",
            #         history=False,
            #     )

            # t = timesteps[i]
            # s = timesteps[i + 1]
            if decoding == "backward":
                # 뒤에 있는 mask 먼저 채우기 하나씩만
                x0 = torch.argmax(mask_logits, dim=-1)
                true_idx = mask_index.nonzero(as_tuple=True)  # 실제 위치 튜플
                x[true_idx[0][-1], true_idx[1][-1]] = x0[-1]
            elif decoding == "forward":
                x0 = torch.argmax(mask_logits, dim=-1)
                true_idx = mask_index.nonzero(as_tuple=True)  # 실제 위치 튜플
                x[true_idx[0][0], true_idx[1][0]] = x0[0]
            elif decoding == "random":
                x0 = torch.argmax(mask_logits, dim=-1)
                true_idx = mask_index.nonzero(as_tuple=True)  # 실제 위치 튜플
                rand_idx = torch.randint(0, true_idx[0].shape[0], (1,))
                x[true_idx[0][rand_idx], true_idx[1][rand_idx]] = x0[rand_idx]


            # if alg == 'origin':
            #     p_transfer = 1 - s / t if i < steps - 1 else 1
            #     x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
            #     transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
            #     _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
            #     print(x0)
            #     x[mask_index] = x0.clone()
            # else:
            #     if alg == 'maskgit_plus':
            #         confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            #     elif alg == 'topk_margin':
            #         confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            #     elif alg == 'entropy':
            #         confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            #     else:
            #         raise RuntimeError(f"Unknown alg: {alg}")
            #     num_mask_token = mask_index.sum() / mask_index.shape[0]
            #     number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            #     full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
            #     full_confidence[mask_index] = confidence
            #     if number_transfer_tokens > 0:
            #         if alg_temp is None or alg_temp == 0:
            #             _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
            #         else:
            #             full_confidence = full_confidence / alg_temp
            #             full_confidence = F.softmax(full_confidence, dim=-1)
            #             transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
            #         x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
            #         x_[mask_index] = x0.clone()
            #         row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
            #         x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            # this allows user-defined token control of the intermediate steps
            # x = generation_tokens_hook_func(i, x, logits)
            # print(x)
            if histories is not None:
                histories.append(x.clone())
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

class DreamWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Dream base model

    # peft가 찾는 생성 훅(통과용)
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    # ★ 핵심: forward 위임
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # 속성 위임(가끔 필요)
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate responses using LLaDA model with optional defense")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--attack_prompt", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--mask_counts", type=int, default=DEFAULT_MASK_COUNTS)
    parser.add_argument("--attack_method", type=str, default="zeroshot")
    parser.add_argument("--defense_method", type=str, default=None)
    parser.add_argument("--decoding", type=str, default="low_confidence")
    return parser.parse_args()


def process_prompt_instruct(prompt: str, mask_counts: int) -> Tuple[str, int]:
    """Replace <mask:x> with repeated <|mask|> and pad if necessary."""
    def replace(match):
        return "<|mask|>" * int(match.group(1))

    prompt = re.sub(SPECIAL_TOKEN_PATTERN, replace, prompt)

    if "<|mask|>" not in prompt and mask_counts:
        prompt += "<|mask|>" * mask_counts

    return prompt, prompt.count("<|mask|>")


def prepare_prompt(tokenizer, behavior: str, is_instruct: bool) -> str:
    """Prepare prompt based on whether model is Instruct-style."""
    if is_instruct:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": behavior}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return behavior


def generate_response(
    vanilla_prompt: str,
    prompt: str,
    tokenizer,
    model,
    args
) -> str:
    """Generate a response given a prompt using diffusion-style generation."""
    input_ids = tokenizer(prompt)["input_ids"]
    attention_mask = tokenizer(prompt)["attention_mask"]

    vanilla_input_ids = tokenizer(vanilla_prompt)["input_ids"]
    vanilla_prompt_len = len(vanilla_input_ids)

    matching_count = next(
        (i for i, (a, b) in enumerate(zip(vanilla_input_ids, input_ids)) if a != b),
        min(len(vanilla_input_ids), len(input_ids))
    )

    mask_indices = [i for i, token in enumerate(input_ids) if token == args.mask_id]

    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(model.device).float()

    if args.decoding == "low_confidence":
        output_ids = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=args.steps,
            temperature=0.2,
            top_p=0.95,
            alg="entropy"
            # decoding=args.decoding,
        ).sequences
    else:
        output_ids = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=args.steps,
            temperature=0.2,
            top_p=0.95,
            decoding=args.decoding,
        ).sequences

    response = tokenizer.batch_decode(output_ids[:, matching_count:], skip_special_tokens=True)[0]

    mask_tokens = [tokenizer.decode(output_ids[0][i]) for i in mask_indices]
    mask_tokens = "".join(mask_tokens)
    if args.attack_method == "DIJA":
        response = response.split("assistant\n")[0]

    return response, mask_tokens


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Initialize defender if needed
    defender = Defender(args.defense_method) if args.defense_method and args.defense_method.lower() != "none" else None

    # Determine instruct model
    is_instruct_model = "Instruct" in args.model_path or "1.5" in args.model_path

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device).eval()
    if args.decoding != "low_confidence":
        model.diffusion_generate = DreamGenerationMixin.diffusion_generate.__get__(model)
        model._sample = DreamGenerationMixin._sample.__get__(model)

    model = DreamWrapper(model)

    lora_path = args.lora_path

    if args.lora_path != "None":
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    else:
        pass

    # Load attack prompts
    with open(args.attack_prompt, 'r') as f:
        data = json.load(f)

    results = []

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data[:80], desc="Processing data")):
            behavior_id = item['BehaviorID']
            vanilla_behavior = item['Behavior']
            refined_behavior = item.get('Refined_behavior', '')

            # Choose behavior
            if args.attack_method == "DIJA" or "refine" in args.attack_prompt:
                behavior = refined_behavior
            else:
                behavior = vanilla_behavior

            # Prepare prompts
            prompt = prepare_prompt(tokenizer, behavior, is_instruct_model)
            vanilla_prompt = prepare_prompt(tokenizer, vanilla_behavior, is_instruct_model)

            # Process masking
            if args.attack_method == "DIJA":
                prompt, _ = process_prompt_instruct(prompt, args.mask_counts)

            # Apply defense if specified
            if defender:
                prompt = defender.handler(prompt)
                vanilla_prompt = defender.handler(vanilla_prompt)

            response, selected_responses = generate_response(vanilla_prompt, prompt, tokenizer, model, args)

            # logging.info(f"{COLOR_BLUE}Response: {selected_responses}{COLOR_RESET}\n")
            print(f"Response: {selected_responses}\n")

            # Save result
            results.append({
                "BehaviorID": behavior_id,
                "Behavior": item['Behavior'],
                "Refined_behavior": refined_behavior,
                "response": response,
                "selected_responses": selected_responses
            })

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
