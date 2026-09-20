"""
Inference with the GRPO-fine-tuned LoRA adapter.

This script is meant to be run in the same Python session that executed
`FineTuning_train.py` (it reuses the `model` and `train_dataset` objects from
the training script) or after restoring those objects from the saved LoRA
checkpoint "grpo_lora".

It generates one answer for a sample from the training set using the loaded
LoRA adapter.
"""

from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=1.0,
    top_k=50,
    max_tokens=1024,
)

outputs = model.fast_generate(
    {
        "prompt": train_dataset[165]["prompt"],
        "multi_modal_data": {"image": train_dataset[165]["image"]},
    },
    sampling_params,
    lora_request=model.load_lora("grpo_lora"),
)
print(outputs[0].outputs[0].text)
