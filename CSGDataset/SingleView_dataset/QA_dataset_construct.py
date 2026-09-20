"""
Single-view QA dataset construction via Zhipu GLM-4.6V-Flash API.

For each street-view image, asks the VLM to generate a structured set of
multiple-choice questions across 6 categories:
  - EC (Entity Count), AD (Attribute Description), SD (Spatial Description),
    Text, CS (Counting/Semantic), RC (Relation Composition).

Outputs one JSON file per image containing sample_id and a questions list.

Usage:
    python QA_dataset_construct.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGES_DIR = Path("data/SingleView_Images")
OUTPUT_DIR = Path("data/QA_Dataset")
LOG_DIR = Path("logs")
API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "GLM-4.6V-Flash"
MAX_RETRIES = 3
LIMIT = 0
IMAGE_MAX_SIDE = 2048

# Chinese prompt for best VLM performance
SYSTEM_PROMPT = """
你是城市场景视觉问答专家。请仔细观察输入的街景图片，生成一组结构完整、逻辑严谨的多选题，用于评估视觉模型对城市静态场景的理解能力。

### 一、任务要求
1.  必须从以下6个类别分别生成题目，每个类别至少1道，总计8-12道题：
    - EC（实体计数类）：统计图中某类静态实体的数量（如建筑数量、路灯数量、树木数量），选项为不同整数
    - AD（属性描述类）：询问实体的外观/功能/属性（如某建筑的颜色、材质、功能用途）
    - SD（空间描述类）：询问实体间的空间位置关系（如A在B的哪一侧、哪个方向）
    - Text（文字识别类）：识别图中招牌、路牌、标识上的文字内容，选项为不同文字片段
    - CS（常识语义类）：基于场景常识判断实体功能/用途（如这个设施的作用是什么）
    - RC（关系组合类）：综合两个以上实体的空间+功能关系进行推理判断
2.  每道题必须包含：
    - question：清晰明确的问题文本
    - options：4个选项，必须为[A]、[B]、[C]、[D]格式，其中1个正确，3个干扰项
    - answer_label：正确选项的标签（A/B/C/D）
    - answer_text：正确选项的完整文本
    - category：所属类别（EC/AD/SD/Text/CS/RC）
3.  仅围绕图片中**清晰可见的静态实体**出题（建筑、道路、交通设施、绿化、商铺、路灯、围栏等），禁止针对行人、车辆、骑手等动态对象出题
4.  干扰项必须合理且具有迷惑性，不能明显错误；正确答案必须唯一且无歧义

### 二、输出格式
严格输出JSON格式，结构如下：
{
  "questions": [
    {
      "question": "问题文本",
      "options": [
        {"label": "A", "text": "选项A内容"},
        {"label": "B", "text": "选项B内容"},
        {"label": "C", "text": "选项C内容"},
        {"label": "D", "text": "选项D内容"}
      ],
      "answer_label": "A",
      "answer_text": "选项A内容",
      "category": "EC"
    }
  ]
}

### 三、质量要求
1.  所有题目必须完全基于图片内容，不得编造画面中不存在的实体
2.  题目难度适中，覆盖场景的核心视觉信息
3.  禁止输出任何额外说明文字，仅输出上述JSON对象
""".strip()


def build_prompt(sample_id: str) -> str:
    return f"样本编号：{sample_id}\n\n请根据图片生成8-12道多选题。只输出JSON。"


def image_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def invoke_with_image(client, system_prompt, user_prompt, image_path, temperature=0.3):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        ]},
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        max_tokens=4096, temperature=temperature, top_p=0.8,
    )
    return resp.choices[0].message.content


def parse_json_response(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty response")
    cleaned = text.strip().lstrip("\ufeff")
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(cleaned):
            if ch == "{":
                try:
                    data, _ = decoder.raw_decode(cleaned[i:])
                    break
                except json.JSONDecodeError:
                    continue
        else:
            raise ValueError("No valid JSON found")
    if not isinstance(data, dict):
        raise ValueError("JSON root must be object")
    return data


def validate_qa(data: dict, sample_id: str) -> dict:
    questions = data.get("questions", [])
    if not isinstance(questions, list) or len(questions) < 6:
        raise ValueError(f"Expected >=6 questions, got {len(questions) if isinstance(questions, list) else 'N/A'}")
    valid_cats = {"EC", "AD", "SD", "Text", "CS", "RC"}
    for q in questions:
        if not all(k in q for k in ("question", "options", "answer_label", "answer_text", "category")):
            raise ValueError("Question missing required fields")
        if q["category"] not in valid_cats:
            raise ValueError(f"Invalid category: {q['category']}")
        if len(q["options"]) != 4:
            raise ValueError(f"Expected 4 options, got {len(q['options'])}")
    return {"sample_id": sample_id, "questions": questions}


def is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return isinstance(d, dict) and len(d.get("questions", [])) >= 6
    except (OSError, json.JSONDecodeError):
        return False


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if ZhipuAiClient is None:
        raise RuntimeError("Install zai: pip install zai")
    if API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("Set ZHIPU_API_KEY environment variable")
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")

    client = ZhipuAiClient(api_key=API_KEY)
    images = sorted(p for p in IMAGES_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if LIMIT > 0:
        images = images[:LIMIT]

    pending = [p for p in images if not is_valid(OUTPUT_DIR / f"{p.stem}.json")]
    print(f"Total: {len(images)}, to process: {len(pending)}, skipped: {len(images) - len(pending)}")

    failed = []
    for i, img_path in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {img_path.name}")
        last_err = ""
        result = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                prompt = build_prompt(img_path.stem)
                if last_err:
                    prompt += f"\n\nPrevious validation failed: {last_err}. Please fix."
                raw = invoke_with_image(client, SYSTEM_PROMPT, prompt, img_path)
                result = validate_qa(parse_json_response(raw), img_path.stem)
                break
            except Exception as exc:
                last_err = str(exc)
                print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {last_err}")
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 8))
        if result is None:
            failed.append({"image": img_path.name, "error": last_err})
        else:
            write_json_atomic(OUTPUT_DIR / f"{img_path.stem}.json", result)

    write_json_atomic(LOG_DIR / "failed_qa_singleview.json", {"failed": failed})
    print(f"\nDone. Success: {len(pending) - len(failed)}, Failed: {len(failed)}")


if __name__ == "__main__":
    main()
