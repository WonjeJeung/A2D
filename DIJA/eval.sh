#!/bin/bash
decoding=low_confidence # low_confidence, forward, backward, random
ft_method=a2d
model_name=llada_1.5
benchmark_name=harmbench

lora_path=../model/${model_name}_${ft_method}
attack_type=DIJA

# Navigate to working directory for model inference
cd $HOME/A2D/DIJA || { echo "Failed to change directory to run_harmbench"; exit 1; }

# Define paths based on model name
if [[ "$model_name" == *"llada_instruct"* ]]; then
    model_path="GSAI-ML/LLaDA-8B-Instruct"  # TODO: Update this path
    python_script="models/harmbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"llada_1.5"* ]]; then
    model_path="GSAI-ML/LLaDA-1.5" # TODO: Update this path
    python_script="models/harmbench_llada.py"
    steps=128
    gen_length=128
    mask_id=126336
    mask_counts=36
elif [[ "$model_name" == *"dream_instruct"* ]]; then
    model_path="Dream-org/Dream-v0-Instruct-7B"  # TODO: Update this path
    python_script="models/harmbench_dream.py"
    steps=64
    gen_length=64
    mask_id=151666
    mask_counts=36
else
    echo "Unknown or unsupported model name: $model_name"
    exit 1
fi

# Define prompt and output file paths
attack_prompt="$HOME/A2D/DIJA/refine_prompt/${attack_type}_${benchmark_name}.json"
output_json="$HOME/A2D/DIJA/attack_results/${attack_type}_${benchmark_name}_${model_name}_${ft_method}_${decoding}.json"


# TODO: Run the jailbreak attack
echo "Running model inference with ${python_script}..."
python ${python_script} \
  --lora_path "${lora_path}" \
  --model_path "${model_path}" \
  --attack_prompt "${attack_prompt}" \
  --output_json "${output_json}" \
  --steps ${steps} \
  --gen_length ${gen_length} \
  --mask_id ${mask_id} \
  --mask_counts ${mask_counts} \
  --attack_method "${attack_type}" \
  --decoding "${decoding}"

# Define evaluation paths
cls_path="cais/HarmBench-Llama-2-13b-cls"
completions_path="${output_json}"
save_path="$HOME/A2D/DIJA/eval_results/eval_results_${attack_type}_${benchmark_name}_${model_name}_${ft_method}_${decoding}.json"

# TODO: Run ASR-e evaluation
echo "Running HarmBench ASR-e..."
python eval_metric/evaluate_completions_asr_e.py \
    --cls_path "${cls_path}" \
    --completions_path "${completions_path}" \
    --save_path "${save_path}"


echo "All steps completed successfully."