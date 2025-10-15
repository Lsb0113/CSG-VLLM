from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 1.0,
    top_k = 50,
    max_tokens = 1024,
)

outputs = model.fast_generate(
    {
        "prompt": train_dataset[165]["prompt"],
        "multi_modal_data": {"image": train_dataset[165]["image"]}
    },
    sampling_params,
    lora_request = model.load_lora("grpo_lora"))
print(outputs[0].outputs[0].text)