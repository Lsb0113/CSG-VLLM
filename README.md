# CSGVLM: City Scene Graphs with Vision-Language Models

A framework for generating and evaluating **semantic 2D scene graphs** from urban street-view imagery using vision-language models (VLMs). The project supports **single-view**, **sequential (order-view)**, and **multi-view (panoramic)** inputs, and provides tools for QA dataset construction, land-use classification, scene-graph QA evaluation, and GRPO-based fine-tuning of VLMs.

![Pipeline](pipeline.png)

## Table of Contents

- [News](#news)
- [Dataset](#dataset)
- [Dependencies & Installation](#dependencies--installation)
- [Project Structure](#project-structure)
- [Scene Graph Generation](#scene-graph-generation)
  - [API-based generation (GLM-4.6V-Flash)](#api-based-generation-glm-46v-flash)
  - [Local-model generation (Qwen3-VL-8B)](#local-model-generation-qwen3-vl-8b)
- [Dataset Construction](#dataset-construction)
  - [QA dataset construction](#qa-dataset-construction)
  - [Land-use annotation](#land-use-annotation)
- [Evaluation](#evaluation)
  - [Scene graph metrics](#scene-graph-metrics)
  - [Scene-graph QA](#scene-graph-qa)
  - [Land-use prediction](#land-use-prediction)
- [Fine-tuning (GRPO)](#fine-tuning-grpo)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)

---

## News

- **2026-09**: All Jupyter notebooks have been converted to standalone `.py` scripts for easier reproduction; comments were cleaned and translated to English. A `requirements.txt` and `.gitignore` were added.

---

## Dataset

The dataset (street-view images in three view configurations, plus ground-truth scene graphs, QA annotations, and land-use labels) is **not bundled** with this repository due to size. The download link is provided in [`Dataset_link.txt`](Dataset_link.txt):

```
https://zenodo.org/records/22855691?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM2ZmZjNTBjLTBkYjMtNDI5NC04YjNkLTc0ZDY3YWZhNjFjNiIsImRhdGEiOnt9LCJyYW5kb20iOiI0MmEwZmQyODQ3ZTMwMzYyZWYxY2M4NzQ3NmU2N2UzNiJ9.VWxDDvMj64N_m0yvsqgEoBZZ-QcnA1J8RpdBQsQv3TexTcbV51lPceBZ7QbZO5-pnuQpcpsky_DHrELkCMNVRw
```

> **Note:** This URL is a Zenodo *preview* link. If it expires or asks for a token, please publish the Zenodo record and update the link in `Dataset_link.txt` (and in this README) to the public record page.

After downloading, place the images so that each script's `IMAGES_DIR` (see configuration tables below) points to the correct folder.

### View configurations

| View Type | Description | Frames per Sample | File Naming |
|-----------|-------------|-------------------|-------------|
| **SingleView** | One perspective street-view image per location | 1 | `<location>.jpg` |
| **OrderView** | Three sequential frames along a road (~0 m / 30 m / 60 m) | 3 | 3 images per location directory |
| **MultiView** | Four panoramic directions (left / front / right / back = 0° / 90° / 180° / 270°) | 4 | `<location>_<heading>_0.jpg` (e.g. `_90_0.jpg`) |

### Scene graph schema

Each generated scene graph is a JSON object with two lists:

```json
{
  "nodes": [
    {
      "id": 1,
      "type": "道路",
      "type_id": 15,
      "feature": "灰色沥青路面，宽阔平整",
      "function": "车辆与行人通行",
      "text": ""
    }
  ],
  "relations": [
    {
      "from": 1,
      "to": 2,
      "Spatial_relation": "上方",
      "Semantic_relation": "照明",
      "Topological_relation": "附着"
    }
  ]
}
```

- **88 entity categories** — buildings, roads, traffic infrastructure, vegetation, street furniture, etc. (see `ENTITY_ID_DICT` in any generation/eval script).
- **26 spatial relations** — left / right / front / back / above / below / corner / center, etc.
- **23 topological relations** — adjacent / surrounds / belongs-to / contains / embedded / supported-by, etc.

> Entity types and relations use **Chinese labels** because the VLMs are prompted in Chinese; this keeps the model output directly mappable to the vocabulary dictionaries. English annotations are provided in code comments.

### Land-use classification (7 classes)

| Class | Description |
|-------|-------------|
| Residential Land | Housing, apartments, residential compounds |
| Highway Land | Expressways, urban expressways, elevated roads, interchanges |
| Urban and Rural Road Land | Ordinary urban / rural roads, intersections |
| Commercial and Service Land | Malls, shops, offices, hotels, restaurants |
| Park and Green space | Urban parks, plazas, lawns, road green belts |
| Forest Land | Natural dense woodland, hillside forests |
| Industrial land | Factories, warehouses, industrial parks, construction sites |

### QA question categories (6 types)

| Code | Category | Description |
|------|----------|-------------|
| EC | Entity Count | Presence / counting of visible static entities |
| AD | Attribute Description | Color, shape, type, material of entities |
| SD | Spatial Description | Relative positions (left/right/front/back, etc.) |
| Text | Text Recognition | Visible text on signs, billboards, street name plates |
| CS | Comparison / Sorting | Compare counts, sizes, or ordering along a road |
| RC | Reasoning / Common Sense | One- or two-step inference from visible evidence |

---

## Dependencies & Installation

Python 3.10+ is recommended.

### Core packages

```bash
pip install -r requirements.txt
```

This installs the common dependencies:

```text
pillow>=10.0
numpy>=1.24
pandas>=2.0
tqdm>=4.66
zai>=0.1.0        # Zhipu AI (GLM) SDK for API-based generation
```

### API key (GLM / Zhipu AI)

All API-based scripts read the key from the `ZHIPU_API_KEY` environment variable. Never hard-code or commit your key.

```bash
# Linux / macOS
export ZHIPU_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:ZHIPU_API_KEY = "your-api-key-here"
```

### Local model (Qwen3-VL-8B)

```bash
pip install torch transformers accelerate
```

- A GPU with **at least 16 GB VRAM** is recommended for `Qwen3-VL-8B` in bfloat16.
- Set `MODEL_PATH` in the script to your local model path or a Hugging Face repo id (default `Qwen3-VL-8B`).
- The scripts disable flash/memory-efficient SDP backends and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid GPU hangs; adjust `CUDA_VISIBLE_DEVICES` as needed (default `0`).

### Fine-tuning (optional)

```bash
pip install unsloth trl datasets vllm
```

---

## Project Structure

```
CSGVLM/
├── README.md
├── pipeline.png
├── requirements.txt
├── .gitignore
├── Dataset_link.txt
├── FineTuning_train.py           # GRPO fine-tuning (unsloth)
├── FineTuning_inference.py       # Inference with the fine-tuned LoRA
│
├── SceneGraph_Generation/
│   ├── API/                      # Zhipu GLM-4.6V-Flash based
│   │   ├── CSG_SingleView.py     # Single-view scene graph
│   │   ├── CSG_MultiView.py      # Multi-view (4 panoramic views) scene graph
│   │   └── CSG_OrderView.py      # Order-view (3 sequential frames) scene graph
│   └── LocalModel/               # Local Qwen3-VL-8B based
│       ├── Qwen_SingleView.py
│       ├── Qwen_MultiView.py
│       └── Qwen_OrderView.py
│
├── CSGDataset/                   # Dataset construction (via GLM API)
│   ├── SingleView_dataset/
│   │   ├── QA_dataset_construct.py      # QA questions from single images
│   │   └── Landtype_SingleView.py       # Land-use labels from single images
│   ├── OrderView_dataset/
│   │   ├── QA_dataset_construct_OrderView.py
│   │   └── landtype_order.py
│   └── MultiView_dataset/
│       ├── QA_dataset_construct_MultiView.py
│       └── Landtype_multiview.py
│
└── Eval/
    ├── SceneGraph_Eval.py        # Scene graph structure / triplet metrics
    ├── SceneGraph_QA_Eval.py     # VQA accuracy from scene graphs
    └── Landtype_prediction.py    # Land-use prediction & evaluation
```

---

## Scene Graph Generation

### API-based generation (GLM-4.6V-Flash)

All scripts under `SceneGraph_Generation/API/` send street-view images to the Zhipu **GLM-4.6V-Flash** model and parse the returned JSON scene graph. Outputs that already exist as non-empty JSON are automatically skipped, so runs are **resumable**.

#### Single view

```bash
cd SceneGraph_Generation/API
python CSG_SingleView.py
```

**Configuration** (top of `CSG_SingleView.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGES_DIR` | `Datasets/perspective_1024/2021/nanshan` | Input image directory |
| `OUTPUT_DIR` | `Outputs/GLM/CSG_singleview` | Output JSON directory |
| `MAX_IMAGES` | `0` | Process only the first N images (0 = all) |
| `MODEL_NAME` | `GLM-4.6V-Flash` | Zhipu model name |

#### Multi view (4 panoramic directions)

```bash
python CSG_MultiView.py
```

**Configuration** (top of `CSG_MultiView.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGES_DIR` | `Datasets/MultiViewDataset_new` | Input directory; each subdirectory = one location with 4 views |
| `OUTPUT_DIR` | `Outputs/GLM/MultiViews_new` | Output JSON directory |
| `MODEL_NAME` | `GLM-4.6V-Flash` | Zhipu model name |

#### Order view (3 sequential frames)

```bash
python CSG_OrderView.py
```

**Configuration** (top of `CSG_OrderView.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGES_DIR` | `Datasets/OderDatasets_interval_2` | Input directory; each subdirectory = one location with 3 frames |
| `OUTPUT_DIR` | `Outputs/GLM/OrderViews_new` | Output JSON directory |
| `MODEL_NAME` | `GLM-4.6V-Flash` | Zhipu model name |

### Local-model generation (Qwen3-VL-8B)

The scripts under `SceneGraph_Generation/LocalModel/` run a locally hosted `Qwen3-VL-8B` model. Generation is deterministic (greedy decoding, `temperature=0.0`).

```bash
cd SceneGraph_Generation/LocalModel
python Qwen_SingleView.py    # single images
python Qwen_MultiView.py     # 4 panoramic views per location
python Qwen_OrderView.py     # 3 sequential frames per location
```

**Common configuration** (top of each script):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Qwen3-VL-8B` | Local model path or Hugging Face repo id |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU index (overridable via environment) |
| `IMAGES_DIR` | per script | Input images (see table below) |
| `OUTPUT_DIR` | per script | Output JSON directory |

| Script | Default `IMAGES_DIR` | Default `OUTPUT_DIR` |
|--------|----------------------|----------------------|
| `Qwen_SingleView.py` | `nanshan_seg_jpeg` | `Outputs/Qwen3/SingleView_typeID` |
| `Qwen_MultiView.py` | `Datasets/MultiViews_Dataset` | `Outputs/Qwen3/MultiViews` |
| `Qwen_OrderView.py` | `Datasets/OrderViewForward_Dataset` | `Outputs/Qwen3/OrderViewForward` |

> **Image-order caution:** in `CSG_MultiView.py` and `Qwen_MultiView.py` the four images are opened in the original notebook order `[front(90°), back(270°), left(0°), right(180°)]`, while the generation prompt describes the four slots as `left / front / right / back`. If your results look inconsistent, align `VIEW_FILES` with the prompt order before regenerating.

---

## Dataset Construction

These scripts build the three auxiliary datasets from raw images using the GLM API. All of them:

- read the API key from `ZHIPU_API_KEY`;
- skip samples whose output JSON already exists and is valid (resumable);
- write a `failed_*.json` report under `logs/` when samples fail after retries.

### QA dataset construction

| Script | Input | Output | Default paths |
|--------|-------|--------|---------------|
| `CSGDataset/SingleView_dataset/QA_dataset_construct.py` | Single images | QA JSON files (8–12 MCQs per image) | `data/SingleView_Images` → `data/QA_Dataset` |
| `CSGDataset/OrderView_dataset/QA_dataset_construct_OrderView.py` | 3-frame sequences | QA JSON files (1 question per category per location) | `data/OrderView_Images` → `data/QA` |
| `CSGDataset/MultiView_dataset/QA_dataset_construct_MultiView.py` | 4-direction views | QA JSON files | `data/MultiViews_Images` → `data/QA` |

Each question records `qa_id`, `category` (`EC` / `AD` / `SD` / `Text` / `CS` / `RC`), `difficulty`, `question`, 4 options, `answer_label`, `answer_text`, `evidence`, and `confidence`.

```bash
cd CSGDataset/SingleView_dataset
python QA_dataset_construct.py
```

### Land-use annotation

| Script | Input | Output | Default paths |
|--------|-------|--------|---------------|
| `CSGDataset/SingleView_dataset/Landtype_SingleView.py` | Single images | Land-use labels | `data/SingleView_Images` → `data/LandType` |
| `CSGDataset/OrderView_dataset/landtype_order.py` | 3-frame sequences | Land-use labels | `data/OrderView_Images` → `data/LandType` |
| `CSGDataset/MultiView_dataset/Landtype_multiview.py` | 4-direction views | Land-use labels | `data/MultiViews_Images` → `data/LandType` |

Each output JSON contains `location_id`, `land_type` (one of the 7 classes), `reason`, `confidence`, and `alternatives`.

```bash
cd CSGDataset/MultiView_dataset
python Landtype_multiview.py
```

---

## Evaluation

### Scene graph metrics

`Eval/SceneGraph_Eval.py` compares predicted scene graphs against ground truth and computes:

- **Entity accuracy** — fraction of ground-truth entity categories correctly predicted;
- **Structure accuracy** — fraction of ground-truth entity-pair structures recovered;
- **Triplet accuracy** — spatial, topological, and overall (spatial ∪ topological) triplet accuracy;
- **Weighted recall** — entity / spatial / topological / relation weighted recall.

```bash
cd Eval
python SceneGraph_Eval.py
```

**Configuration** (edit the `__main__` block):

```python
evaluator.run(
    gt_dir="../Datasets/SingleView_constrain_classes_15",  # ground-truth JSON dir
    pred_dir="../Datasets/SingleView_constrain_classes",   # predicted JSON dir
    csv_name="result.csv",
)
```

**Outputs:**
- `result.csv` — per-sample metrics (entity / structure / triplet accuracies, overall);
- Console summary with entity / spatial / topology weighted recall.

### Scene-graph QA

`Eval/SceneGraph_QA_Eval.py` sends the structured scene graph (**text only**, no image) plus a multiple-choice question to a VLM and computes per-category and overall accuracy for the 6 question categories.

```bash
cd Eval
python SceneGraph_QA_Eval.py \
    --qa-dir StreetQADataset_v1 \
    --sg-dir Outputs/GLM/CSG_singleview \
    --output-dir QA_experiments/run_01 \
    --api-key YOUR_KEY          # or set ZHIPU_API_KEY
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--qa-dir` | `StreetQADataset_v1` | Ground-truth QA JSON directory |
| `--sg-dir` | `Outputs/GLM/BSG_QA_v1` | Predicted scene graph JSON directory |
| `--output-dir` | `QA_experiments/BSG_QA_GLM` | Results output directory |
| `--api-key` | `ZHIPU_API_KEY` env | Zhipu API key |
| `--model-name` | `GLM-4-Flash` | VLM used to answer the questions |
| `--limit` | none | Debug: process only the first N samples |

**Outputs:**
- `scene_graph_qa_results.csv` — per-question results (gold vs. predicted label, raw response);
- `scene_graph_qa_details.json` — full details;
- Console summary with per-category accuracy (EC / AD / SD / Text / CS / RC) and overall accuracy.

### Land-use prediction

`Eval/Landtype_prediction.py` predicts the dominant land-use category from a scene graph (text only) using a VLM, then evaluates the predictions against ground-truth labels.

```bash
cd Eval
python Landtype_prediction.py
```

**Configuration** (top of file):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCENEGRAPH_DIR` | `data/OrderView_dataset/SceneGraphs` | Scene graph JSON directory |
| `GROUND_TRUTH_DIR` | `data/OrderView_dataset/LandType` | Ground-truth labels |
| `PREDICTION_DIR` | `data/OrderView_dataset/LandType_Prediction` | Output predictions |
| `MODEL_NAME` | `GLM-4.6V-Flash` | VLM used for prediction |

**Outputs** (written into `PREDICTION_DIR`):
- `prediction_results.csv` — per-scene predictions (with confidence and reasoning);
- `per_class_metrics.csv` — precision / recall / F1 per class;
- `confusion_matrix.csv` — confusion matrix;
- `evaluation_report.json` — overall accuracy, majority baseline, macro / weighted F1, per-class details.

---

## Fine-tuning (GRPO)

`FineTuning_train.py` fine-tunes `unsloth/Qwen2.5-VL-7B-Instruct` with **LoRA + GRPO** on a QA dataset using the `unsloth` / `trl` stack. It expects a Hugging Face dataset named `Data` (split `testmini`) with columns `question`, `answer`, and `decoded_image`; adjust the `load_dataset("Data", ...)` call for your own dataset.

```bash
python FineTuning_train.py
```

Key settings: `lora_rank=16`, 4-bit quantization, `GRPOConfig` with GSPO (`loss_type="dr_grpo"`), and two reward functions (formatting + answer correctness). The trained LoRA adapter is saved to `grpo_lora/`.

`FineTuning_inference.py` generates one answer with the fine-tuned adapter using vLLM fast inference (run it in the same session as the training script, or restore `model` / `train_dataset` from the saved checkpoint first).

---

## Reproducibility Notes

- **Images are not included** in this repository. Download the dataset from the link in `Dataset_link.txt`, then update each script's `IMAGES_DIR` (or equivalent) to point at your local data.
- All API scripts require `ZHIPU_API_KEY`; local-model scripts require `MODEL_PATH` (default `Qwen3-VL-8B`) and a GPU with ≥ 16 GB VRAM.
- All generation / construction scripts are **resumable**: samples with a non-empty output JSON are skipped automatically.
- Generation is **deterministic** for local models (`do_sample=False`, `temperature=0.0`, `num_beams=1`); API calls use fixed temperature / top-p values documented in each script.
- Entity / relation / land-use vocabularies use Chinese labels because the VLMs are prompted in Chinese; the corresponding English names are provided in comments and in the tables above.
- Output directories (`Datasets/`, `data/`, `Outputs/`, `logs/`) are gitignored — see `.gitignore`.

---

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@misc{csgvlm2026,
  title  = {CSGVLM: City Scene Graphs with Vision-Language Models},
  author = {TODO: add your name(s)},
  year   = {2026},
  howpublished = {\url{https://github.com/TODO/CSG-VLLM}}
}
```

*(Update the author and repository URL before publishing.)*
