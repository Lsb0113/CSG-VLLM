"""
Order-view QA dataset construction via Zhipu GLM-4.6V-Flash API.

For each group of 3 sequential street frames (0m/30m/60m along a road),
asks the VLM to generate multiple-choice questions across 6 categories
(EC/AD/SD/Text/CS/RC), with optional self-review.

Usage:
    python QA_dataset_construct_OrderView.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
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
IMAGES_DIR = Path("data/OrderView_Images")
OUTPUT_DIR = Path("data/QA")
LOG_DIR = Path("logs")
API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "GLM-4.6V-Flash"
FRAMES_PER_GROUP = 3
MAX_RETRIES = 3
ENABLE_REVIEW = False
LIMIT = 0
IMAGE_MAX_SIDE = 2048

FRAME_POSITIONS = {1: "0m (start)", 2: "~30m ahead", 3: "~60m ahead"}
HEADING_RE = re.compile(r"_(\d{1,3})_0$")

QUESTION_CATEGORIES = ("EC", "AD", "SD", "Text", "CS", "RC")
DIFFICULTY_BY_CATEGORY = {
    "EC": "Simple", "AD": "Simple", "SD": "Middle",
    "Text": "Middle", "CS": "Hard", "RC": "Hard",
}

SYSTEM_PROMPT = """
你是城市街景问答数据集标注专家。系统会给你同一段道路上沿行进方向的3帧顺序街景：第1帧约0米、第2帧约30米、第3帧约60米。请把3帧融合理解为同一段连续道路场景，生成高质量选择题。

原则：
1. 只能依据图像中清晰可见的内容，不得猜测画面外信息。
2. 同一物体在相邻帧中先后出现时，计数必须跨帧去重。
3. 每题只有一个明确正确答案。
4. 依赖位置的题目必须写清参照的帧。
5. 计数只统计能明确区分的实例。
6. 文字模糊时不生成文字识别题。
7. 推理题最多两步，只使用普遍常识。
8. 只输出一个合法JSON对象，不要输出Markdown。
""".strip()


def build_generation_prompt(sample_id: str) -> str:
    return f"""
样本编号：{sample_id}

该样本包含3帧顺序街景（图1=0米、图2=30米、图3=60米）。请为以下六种题型各生成1道题：
- EC（存在与计数）：判断实体是否存在或统计数量；跨帧去重。
- AD（属性描述）：询问实体的颜色、形状、类型等直接可见属性。
- SD（空间与方位）：询问左右、前后、沿途先后等空间关系。
- Text（文字识别）：询问招牌、路牌上清晰可读的文字。
- CS（比较与排序）：比较实体的数量、大小或出现先后。
- RC（推理与常识）：根据交通标志、设施等进行一到两步推理。

选项规则：
- 是非题用2个选项；其他题用4-6个选项。
- 选项必须属于相同语义类型，禁止"以上都对""无法判断"。
- 正确答案位置随机分布。

输出结构：
{{
  "sample_id": "{sample_id}",
  "questions": [
    {{
      "qa_id": "{sample_id}_EC_01",
      "category": "EC",
      "difficulty": "Simple",
      "question": "问题文本",
      "options": [
        {{"label": "A", "text": "选项内容"}},
        {{"label": "B", "text": "选项内容"}}
      ],
      "answer_label": "A",
      "answer_text": "正确选项完整文本",
      "evidence": "简短可见证据（注明帧）",
      "confidence": 0.95
    }}
  ],
  "skipped_categories": [
    {{"category": "Text", "reason": "无清晰可读文字"}}
  ]
}}
""".strip()


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qa_orderview")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / "build_qa_order.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def image_to_data_url(image_path: Path) -> str:
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def collect_frames(sample_dir: Path):
    images = sorted(p for p in sample_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != FRAMES_PER_GROUP:
        raise ValueError(f"{sample_dir.name}: expected {FRAMES_PER_GROUP} frames, got {len(images)}")
    return [(i + 1, FRAME_POSITIONS[i + 1], images[i]) for i in range(FRAMES_PER_GROUP)]


def invoke_with_frames(client, system_prompt, user_prompt, frame_items, temperature):
    content = [{"type": "text", "text": "以下是同一段道路的3帧顺序街景图。"}]
    for fno, pos, path in frame_items:
        content.append({"type": "text", "text": f"图{fno}：{pos}"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        max_tokens=8192, temperature=temperature, top_p=0.8,
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


def is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return isinstance(d, dict) and len(d.get("questions", [])) >= 4
    except (OSError, json.JSONDecodeError):
        return False


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def scene_sort_key(path: Path):
    m = re.fullmatch(r"scene_(\d+)", path.name)
    return (0, int(m.group(1))) if m else (1, path.name)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOG_DIR)
    if ZhipuAiClient is None:
        raise RuntimeError("Install zai: pip install zai")
    if API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("Set ZHIPU_API_KEY environment variable")
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")

    client = ZhipuAiClient(api_key=API_KEY)
    sample_dirs = sorted(
        (p for p in IMAGES_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=scene_sort_key,
    )
    if LIMIT > 0:
        sample_dirs = sample_dirs[:LIMIT]

    pending = [d for d in sample_dirs if not is_valid(OUTPUT_DIR / f"{d.name}.json")]
    logger.info("Total: %d, to process: %d", len(sample_dirs), len(pending))

    failed = []
    for i, sample_dir in enumerate(pending, 1):
        logger.info("[%d/%d] %s", i, len(pending), sample_dir.name)
        last_err = ""
        result = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                frames = collect_frames(sample_dir)
                prompt = build_generation_prompt(sample_dir.name)
                if last_err:
                    prompt += f"\n\nPrevious validation failed: {last_err}. Please fix."
                raw = invoke_with_frames(client, SYSTEM_PROMPT, prompt, frames, temperature=0.3)
                result = parse_json_response(raw)
                if "questions" not in result:
                    raise ValueError("Missing 'questions' field")
                break
            except Exception as exc:
                last_err = str(exc)
                logger.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, last_err)
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 8))
        if result is None:
            failed.append({"scene": sample_dir.name, "error": last_err})
            logger.error("%s failed: %s", sample_dir.name, last_err)
        else:
            write_json_atomic(OUTPUT_DIR / f"{sample_dir.name}.json", result)

    write_json_atomic(LOG_DIR / "failed_qa_order.json", {"failed": failed})
    logger.info("Done. Success: %d, Failed: %d", len(pending) - len(failed), len(failed))


if __name__ == "__main__":
    main()
