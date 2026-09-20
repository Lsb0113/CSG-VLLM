"""
Land-use type classification from multi-view (4-direction) street images.

Given 4 panoramic views (left/front/right/back = 0/90/180/270 degrees)
at the same location, sends them to GLM-4.6V-Flash and predicts the
dominant land-use category.

Usage:
    python Landtype_multiview.py
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
IMAGES_DIR = Path("data/MultiViews_Images")
OUTPUT_DIR = Path("data/LandType")
LOG_DIR = Path("logs")
API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "GLM-4.6V-Flash"
MAX_RETRIES = 3
LIMIT = 0
IMAGE_MAX_SIDE = 2048

REQUIRED_ANGLES = (0, 90, 180, 270)
ANGLE_VIEW_NAME = {0: "Left", 90: "Front", 180: "Right", 270: "Back"}
VIEW_SUFFIX_RE = re.compile(r"_(\d{1,3})_0$")

LAND_TYPES = [
    "Residential Land", "Highway Land", "Urban and Rural Road Land",
    "Commercial and Service Land", "Park and Green space",
    "Forest Land", "Industrial land",
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

SYSTEM_PROMPT = """
你是城市街景用地类型判读专家。系统会给你同一地点的4张街景图：左(0°)、前(90°)、右(180°)、后(270°)，覆盖360°环境。请综合4个视角判断该地点最主要的用地类型。

总原则：
- 只能依据图像中清晰可见的内容，不得猜测画面外信息。
- 行人、车辆等动态对象不作为判读依据。
- 用地类型必须且只能从下列7类中选择1类。

7类用地：
1. Residential Land：居民住宅、公寓、住宅小区等。
2. Highway Land：高速公路、城市快速路、高架路、立交桥等。
3. Urban and Rural Road Land：普通城市/乡村道路、交叉口等。
4. Commercial and Service Land：商场、临街商铺、写字楼、酒店等。
5. Park and Green space：城市公园、广场绿地、草坪等。
6. Forest Land：自然茂密乔木、山林等。
7. Industrial land：工厂、仓库、工业园区、在建工地等。

输出要求：只输出一个合法JSON对象。结构：
{
  "location_id": "地点编号",
  "land_type": "7个英文名称之一",
  "reason": "简短判断依据，30至120字",
  "confidence": 0.0,
  "alternatives": ["次可能类别名"]
}
""".strip()


def build_prompt(location_id: str) -> str:
    return f"地点编号：{location_id}\n\n请综合4个视角判断该地点的主导用地类型。只输出JSON。"


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("landtype_multiview")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / "build_landtype_multiview.log", encoding="utf-8")
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


def collect_views(sample_dir: Path):
    angle_to_path = {}
    for p in sorted(sample_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            continue
        m = VIEW_SUFFIX_RE.search(p.stem)
        if m:
            angle_to_path[int(m.group(1))] = p
    missing = [a for a in REQUIRED_ANGLES if a not in angle_to_path]
    if missing:
        raise ValueError(f"{sample_dir.name}: missing angles {missing}")
    return [(a, ANGLE_VIEW_NAME[a], angle_to_path[a]) for a in REQUIRED_ANGLES]


def invoke_with_views(client, system_prompt, user_prompt, view_items, temperature):
    content = [{"type": "text", "text": "以下是同一地点的4张街景图：左/前/右/后。"}]
    for i, (angle, name, path) in enumerate(view_items, 1):
        content.append({"type": "text", "text": f"图{i}：{name} ({angle}°)"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
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


def validate(data: dict, location_id: str) -> dict:
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
    return {"location_id": location_id, "land_type": lt,
            "reason": reason, "confidence": round(conf, 4), "alternatives": alts[:2]}


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
        if p.name in {"landtype_summary.json", "landtype_map.json"}:
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
    write_json_atomic(output_dir / "landtype_map.json",
                      {r.get("location_id", ""): r["land_type"] for r in records})
    logger.info("Aggregated %d locations: %s", len(records), counts)


def process_location(client, sample_dir, view_items, out_path, logger, max_retries):
    loc_id = sample_dir.name
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_prompt(loc_id)
            if last_err:
                prompt += f"\n\nPrevious validation failed: {last_err}. Please fix and re-output JSON."
            raw = invoke_with_views(client, SYSTEM_PROMPT, prompt, view_items, temperature=0.1)
            result = validate(parse_json_response(raw), loc_id)
            result["views"] = [{"view": n, "angle": a, "file": fp.name}
                               for a, n, fp in view_items]
            result["image_files"] = [fp.name for _, _, fp in view_items]
            result["model"] = MODEL_NAME
            result["status"] = "complete"
            write_json_atomic(out_path, result)
            logger.info("%s -> %s (conf=%.2f)", loc_id, result["land_type"], result["confidence"])
            return
        except Exception as exc:
            last_err = str(exc)
            logger.warning("%s attempt %d/%d failed: %s", loc_id, attempt, max_retries, last_err)
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
    sample_dirs = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if LIMIT > 0:
        sample_dirs = sample_dirs[:LIMIT]

    pending = []
    for sd in sample_dirs:
        out = OUTPUT_DIR / f"{sd.name}.json"
        if not is_valid(out):
            pending.append((sd, out))
    logger.info("Total: %d, to process: %d", len(sample_dirs), len(pending))

    failed = []
    for i, (sd, out) in enumerate(pending, 1):
        logger.info("[%d/%d] %s", i, len(pending), sd.name)
        try:
            views = collect_views(sd)
            process_location(client, sd, views, out, logger, MAX_RETRIES)
        except Exception as exc:
            failed.append({"scene": sd.name, "error": str(exc)})
            logger.error("%s failed: %s", sd.name, exc)
    write_json_atomic(LOG_DIR / "failed_landtype_multiview.json", {"failed": failed})
    aggregate(OUTPUT_DIR, logger)


if __name__ == "__main__":
    main()
