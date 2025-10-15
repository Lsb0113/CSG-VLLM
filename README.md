## :book: Construction of unified cognitive scene graphs for city-scale geospatial reasoning
<image src="demo.png" width="100%">

# Introduction
This is a release of the code of our paper **_Construction of unified cognitive scene graphs for city-scale geospatial reasoning_**.

Authors:
XXXXX, 
XXXXX, 
XXXXX, 
XXXXX, 
XXXXX, 
XXXXX, 
XXXXX, 
XXXXX

[[code]](https://github.com/Lsb0113/CSG-VLLM)

# Dependencies
```bash
conda create -n DA3SG python=3.8
conda activate DA3SG
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --upgrade huggingface_hub
pip install unsloth
pip install transformers==4.57.0
pip install --no-deps trl==0.22.2
```

# Experimental environment
Graphics Card: RTX 4090 24GB; CPU: 12 cores; Memory: 60GB; Model: Intel(R) Xeon(R) Gold 6430

# Prepare the data
Download Dataset for fine-tuning, you can follow [Shenzhen Street Map](https://drive.google.com/file/d/1GYjf3sd72etOM--jAN5hjeQqVX0jkvs6/view?usp=drive_link). 

# Fine-tuning Process
A.  You should arrange the file location like this
```
data
  Images
    2190_0900570012210302101537052GH_113.910295198000000_22.535329938899999_0_0.jpg
    ...
  Train_files
    2190_0900570012210302101537052GH_113.910295198000000_22.535329938899999_0_0.json
    ...
...  
      
```

B. Run train.py
```bash
python FineTuning_train.py
```

C. Inference your model 

``` python FineTuning_inference.py ```

or just use the checkpoint 

``` clip_adapter/checkpoint/origin_mean.pth ```

D. You can also directly use our finely-tuned large model. You can download from [Scene Graph Extraction LM]([https://drive.google.com/file/d/1GYjf3sd72etOM--jAN5hjeQqVX0jkvs6/view?usp=drive_link](https://huggingface.co/SHIBIN99/CSG-VLLM)). 


# Run Code
Before running the main file, you can modify some of the experimental settings to suit your needs.
```bash
python main.py
```
In this repo, we have provided a default [config](https://github.com/Lsb0113/ViT3SG/blob/main/config/mmgnet.json)

# Acknowledgement
This repository is partly based on [VL-SAT](https://github.com/wz7in/CVPR2023-VLSAT), [PointMLP](https://github.com/ma-xu/pointMLP-pytorch) and [CLIP](https://github.com/openai/CLIP) repositories.
