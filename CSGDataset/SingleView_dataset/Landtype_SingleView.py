"""
Land-use type classification from single-view street images.

Sends each street-view image to the GLM-4.6V-Flash multimodal model and asks
it to predict the dominant land-use category (7 classes). Outputs one JSON
per image plus aggregated summary files.

Usage:
    python Landtype_SingleView.py
"""

from __future__ import annotations

import base64
import csv
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
IMAGES_DIR = Path("data/SingleView_Images")
OUTPUT_DIR = Path("data/LandType")
LOG_DIR = Path("logs")
API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "GLM-4.6V-Flash"
MAX_RETRIES = 3
LIMIT = 0
IMAGE_MAX_SIDE = 2048

HEADING_RE = re.compile(r"_(\d{1,3})_0$")

LAND_TYPES = [
    "Residential Land",
    "Highway Land",
    "Urban and Rural Road Land",
    "Commercial and Service Land",
    "Park and Green space",
    "Forest Land",
    "Industrial land",
]

LAND_TYPE_CN_ALIAS = {
    "居住用地": "Residential Land", "住宅用地": "Residential Land", "居民用地": "Residential Land",
    "公路用地": "Highway Land", "高速公路用地": "Highway Land", "高速用地": "Highway Land",
    "城市快速路用地": "Highway Land",
    "城乡道路用地": "Urban and Rural Road Land", "道路用地": "Urban and Rural Road Land",
    "城市道路用地": "Urban and Rural Road Land", "乡村道路用地": "Urban and Rural Road Land",
    "商业服务业设施用地": "Commercial and Service Land", "商业服务用地": "Commercial and Service Land",
    "商业用地": "Commercial and Service Land",
    "公园与绿地": "Park and Green space", "公园绿地": "Park and Green space",
    "绿地": "Park and Green space", "公园用地": "Park and Green space",
    "林地": "Forest Land", "森林用地": "Forest Land",
    "工业用地": "Industrial land", "工矿用地": "Industrial land",
}

# Chinese prompt for the VLM (best performance with Chinese instructions)
SYSTEM_PROMPT = """
你是城市街景用地类型判读专家。系统会给你一张街景图像（单个拍摄视角）。请根据这张图中清晰可见的建筑、道路、设施与地表覆盖，判断拍摄地点最主要（主导功能）的用地类型。

总原则：
- 只能依据图像中清晰可见的内容，不得猜测画面外、被遮挡、模糊或无法确认的信息。
- 行人、车辆等动态对象不作为判读依据，以静态的建筑、道路、设施与地表覆盖为准。
- 单张图只呈现一个视角；若证据不足以完全确定，可降低 confidence 并把次可能类型放入 alternatives。
- 用地类型必须且只能从下列7类中选择1类，输出 land_type 时使用括号前的英文名称。

7类用地及判读要点：
1. Residential Land：居民住宅、公寓、住宅小区等。
2. Highway Land：高速公路、城市快速路、高架路、立交桥、匝道等。
3. Urban and Rural Road Land：普通城市/乡村道路、交叉口等。
4. Commercial and Service Land：商场、临街商铺、写字楼、酒店、市场等。
5. Park and Green space：城市公园、广场绿地、草坪、道路绿化带等。
6. Forest Land：自然茂密乔木、山林、郊野森林等。
7. Industrial land：工厂厂房、仓库、工业园区、物流仓储、在建工地等。

输出要求：只输出一个合法JSON对象，不要输出Markdown代码块。结构：
{
  "sample_id": "样本编号",
  "land_type": "上述7个英文名称之一",
  "reason": "简短判断依据，30至120字",
  "confidence": 0.0,
  "alternatives": ["次可能类别名，没有则为空数组"]
}
""".strip()


def build_prompt(sample_id: str) -> str:
    return f"样本编号：{sample_id}\n\n请根据图中清晰可见的内容判断该地点的主导用地类型。只输出JSON。"


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("landtype_singleview")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / "build_landtype_single.log", encoding="utf-8")
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


def detect_heading(image_path: Path):
    m = HEADING_RE.search(image_path.stem)
    return m.group(1) if m else None


def invoke_with_image(client, system_prompt, user_prompt, image_path, temperature):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        ]},
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        max_tokens=1024, temperature=temperature, top_p=0.7,
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


def normalize_land_type(value) -> str:
    if not isinstance(value, str):
        raise ValueError(f"land_type not string: {value!r}")
    text = value.strip().rstrip("。.")
    lower_map = {n.lower(): n for n in LAND_TYPES}
    if text in LAND_TYPES:
        return text
    if text.lower() in lower_map:
        return lower_map[text.lower()]
    if text in LAND_TYPE_CN_ALIAS:
        return LAND_TYPE_CN_ALIAS[text]
    for n in LAND_TYPES:
        if n.lower() in text.lower():
            return n
    raise ValueError(f"Unknown land type: {value!r}")


def validate(data: dict, sample_id: str) -> dict:
    lt = normalize_land_type(data.get("land_type"))
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise ValueError("Missing reason")
    conf = float(data.get("confidence", 0))
    if not 0 <= conf <= 1:
        raise ValueError("confidence out of range")
    alts = []
    for item in data.get("alternatives", []) or []:
        try:
            a = normalize_land_type(item)
        except ValueError:
            continue
        if a != lt and a not in alts:
            alts.append(a)
    return {
        "sample_id": sample_id, "land_type": lt,
        "reason": reason, "confidence": round(conf, 4),
        "alternatives": alts[:2],
    }


def is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return isinstance(d, dict) and d.get("status") == "complete" and d.get("land_type") in LAND_TYPES
    except (OSError, json.JSONDecodeError):
        return False


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def aggregate(output_dir: Path, logger) -> None:
    records = []
    for p in sorted(output_dir.glob("*.json")):
        if p.name == "landtype_summary.json":
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and d.get("land_type") in LAND_TYPES:
            records.append(d)
    counts = {n: 0 for n in LAND_TYPES}
    for r in records:
        counts[r["land_type"]] += 1
    write_json_atomic(output_dir / "landtype_summary.json",
                      {"total": len(records), "counts": counts, "records": records})
    with (output_dir / "landtype_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "land_type", "confidence", "heading", "alternatives", "reason"])
        for r in records:
            w.writerow([r.get("sample_id", ""), r.get("land_type", ""), r.get("confidence", ""),
                        r.get("heading", ""), "; ".join(r.get("alternatives", [])), r.get("reason", "")])
    logger.info("Aggregated %d images: %s", len(records), counts)


def process_image(client, image_path: Path, out_path: Path, logger, max_retries: int) -> None:
    sid = image_path.stem
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_prompt(sid)
            if last_err:
                prompt += f"\n\nPrevious validation failed: {last_err}. Please fix and re-output JSON."
            raw = invoke_with_image(client, SYSTEM_PROMPT, prompt, image_path, temperature=0.1)
            result = validate(parse_json_response(raw), sid)
            result["heading"] = detect_heading(image_path)
            result["image_file"] = image_path.name
            result["model"] = MODEL_NAME
            result["status"] = "complete"
            write_json_atomic(out_path, result)
            logger.info("%s -> %s (conf=%.2f)", image_path.name, result["land_type"], result["confidence"])
            return
        except Exception as exc:
            last_err = str(exc)
            logger.warning("%s attempt %d/%d failed: %s", image_path.name, attempt, max_retries, last_err)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Max retries: {last_err}")


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
    images = sorted(p for p in IMAGES_DIR.iterdir()
                     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if LIMIT > 0:
        images = images[:LIMIT]
    pending = [p for p in images if not is_valid(OUTPUT_DIR / f"{p.stem}.json")]
    logger.info("Total: %d, to process: %d, skipped: %d", len(images), len(pending), len(images) - len(pending))

    failed = []
    for i, p in enumerate(pending, 1):
        logger.info("[%d/%d] %s", i, len(pending), p.name)
        try:
            process_image(client, p, OUTPUT_DIR / f"{p.stem}.json", logger, MAX_RETRIES)
        except Exception as exc:
            failed.append({"image": p.name, "error": str(exc)})
            logger.error("%s failed: %s", p.name, exc)
    write_json_atomic(LOG_DIR / "failed_landtype_single.json", {"failed": failed})
    aggregate(OUTPUT_DIR, logger)


if __name__ == "__main__":
    main()
