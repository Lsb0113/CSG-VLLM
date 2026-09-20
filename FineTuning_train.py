"""
GRPO (Group Relative Policy Optimization) fine-tuning of a Qwen2.5-VL model
with LoRA using the unsloth library.

The script loads a Hugging Face dataset called "Data" (split "testmini"),
filters numeric-answer samples, resizes and converts images, then trains the
vision-language model with GRPO on a reasoning-then-answer format.

Usage:
    python FineTuning_train.py

Requirements:
    pip install unsloth trl datasets torch transformers

Note: the dataset "Data" must be available in the current directory (or
adjust the `load_dataset` call below to your own dataset name / path).
"""

import re

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel

max_seq_length = 16384   # required to be this long for VLMs
lora_rank = 16           # larger rank = smarter, but slower

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,              # False for LoRA 16bit
    fast_inference=True,            # enable vLLM fast inference
    gpu_memory_utilization=0.8,     # reduce if out of memory
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,   # False if not fine-tuning vision layers
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,                           # larger = higher accuracy, may overfit
    lora_alpha=16,                  # recommended alpha == r
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,               # rank-stabilized LoRA
    loftq_config=None,              # LoftQ quantization
    use_gradient_checkpointing="unsloth",  # reduces memory usage
)

dataset = load_dataset("Data", split="testmini")


def is_numeric_answer(example):
    try:
        float(example["answer"])
        return True
    except (TypeError, ValueError):
        return False


dataset = dataset.filter(is_numeric_answer)


def resize_images(example):
    image = example["decoded_image"].resize((512, 512))
    example["decoded_image"] = image
    return example


dataset = dataset.map(resize_images)


def convert_to_rgb(example):
    image = example["decoded_image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["decoded_image"] = image
    return example


dataset = dataset.map(convert_to_rgb)

# Delimiters that wrap the model's reasoning and final answer.
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def make_conversation(example):
    text_content = (
        f"{example['question']}. Also first provide your reasoning or working out"
        f" on how you would go about solving the question between {REASONING_START}"
        f" and {REASONING_END}"
        f" and then your final answer between {SOLUTION_START} and"
        f" (put a single float here) {SOLUTION_END}"
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},                     # placeholder for the image
                {"type": "text", "text": text_content},
            ],
        },
    ]
    return {"prompt": prompt, "image": example["decoded_image"], "answer": example["answer"]}


train_dataset = dataset.map(make_conversation)

# 'decoded_image' holds the actual images, so rename it to the column the
# chat template expects.
train_dataset = train_dataset.remove_columns("image")
train_dataset = train_dataset.rename_column("decoded_image", "image")

train_dataset = train_dataset.map(
    lambda example: {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,   # must add assistant
        )
    }
)


# Reward functions -----------------------------------------------------------
def formatting_reward_func(completions, **kwargs):
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    scores = []
    for completion in completions:
        score = 0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0

        # Penalize completions dominated by repeated "addCriterion" tokens or
        # newlines (see unsloth VLM-RL docs for background).
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion) - len(removal)) / len(completion) >= 0.5:
                score -= 2.0

        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    responses = [re.findall(answer_pattern, c, re.DOTALL) for c in completions]
    q = prompts[0]
    print("-" * 20, f"\nQuestion:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")
    return [
        2.0 if len(r) == 1 and a == r[0].replace("\n", "") else 0.0
        for r, a in zip(responses, answer)
    ]


training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    log_completions=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,   # increase to 4 for smoother training
    num_generations=4,               # decrease if out of memory
    max_prompt_length=1024,
    max_completion_length=1024,
    num_train_epochs=0.5,            # set to 1 for a full training run
    save_steps=60,
    max_grad_norm=0.1,
    report_to="none",                # can use Weights & Biases
    output_dir="outputs",
    # GSPO settings:
    importance_sampling_level="sequence",
    mask_truncated_completions=False,
    loss_type="dr_grpo",
)

if __name__ == "__main__":
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,   # pass the processor for multimodal inputs
        reward_funcs=[
            formatting_reward_func,
            correctness_reward_func,
        ],
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_lora("grpo_lora")
