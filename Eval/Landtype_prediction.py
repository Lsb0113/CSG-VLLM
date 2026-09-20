"""
Land-use type prediction from scene graphs.

Given structured scene graph JSON files (nodes + relations, text only, no images),
this script asks a VLM to predict the dominant land-use category for each location,
then evaluates accuracy against ground-truth labels.

Land-use categories (7 classes):
    Residential Land, Highway Land, Urban and Rural Road Land,
    Commercial and Service Land, Park and Green space, Forest Land, Industrial Land.

Usage:
    python Landtype_prediction.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("data/OrderView_dataset")
SCENEGRAPH_DIR = BASE_DIR / "SceneGraphs"
GROUND_TRUTH_DIR = BASE_DIR / "LandType"
PREDICTION_DIR = BASE_DIR / "LandType_Prediction"
LOG_DIR = BASE_DIR / "logs"

API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "GLM-4.6V-Flash"
MAX_RETRIES = 3
LIMIT = 0  # 0 = all scenes; set to N for quick testing

LAND_TYPES = [
    "Residential Land",
    "Highway Land",
    "Urban and Rural Road Land",
    "Commercial and Service Land",
    "Park and Green space",
    "Forest Land",
    "Industrial land",
]

LAND_TYPE_CN = {
    "Residential Land": "Residential",
    "Highway Land": "Highway",
    "Urban and Rural Road Land": "Urban/Rural Road",
    "Commercial and Service Land": "Commercial/Service",
    "Park and Green space": "Park/Green space",
    "Forest Land": "Forest",
    "Industrial land": "Industrial",
}

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

SHORT_LABEL = {
    "Residential Land": "Residential",
    "Highway Land": "Highway",
    "Urban and Rural Road Land": "Road",
    "Commercial and Service Land": "Commercial",
    "Park and Green space": "Park",
    "Forest Land": "Forest",
    "Industrial land": "Industrial",
}

# Prompt (Chinese for best model performance; explains the 7 land-use classes)
SYSTEM_PROMPT = """
你是城市用地类型判读专家。系统会给你某段道路的【场景图】，其中只包含结构化文本：一组场景节点 nodes（每个节点有类型 type、功能 function、外观特征 feature、招牌/文字 text）以及节点之间的关系 relations。你的任务是只依据场景图信息，推断该地点最主要（主导功能）的用地类型。

总原则：
- 只能依据场景图中明确给出的节点类型、功能、特征、文字和关系，不得引入场景图之外的臆测。
- 重点综合节点的"功能/类型"及其规模与空间关系。
- 行人、车辆等动态对象不作为主导用地依据。
- 多种功能混合时，选择占比最大、最能代表主导功能的一类。
- 用地类型必须且只能从下列7类中选择1类，输出 land_type 时使用括号前的英文名称。

7类用地：
1. Residential Land：居民住宅、公寓、住宅小区等。
2. Highway Land：高速公路、城市快速路、高架路、立交桥、匝道等。
3. Urban and Rural Road Land：普通城市/乡村道路、交叉口等。
4. Commercial and Service Land：商场、临街商铺、写字楼、酒店、市场等。
5. Park and Green space：城市公园、广场绿地、草坪、道路绿化带等。
6. Forest Land：自然茂密乔木、山林、郊野森林等。
7. Industrial land：工厂厂房、仓库、工业园区、物流仓储、在建工地等。

输出要求：只输出一个合法JSON对象，不要输出Markdown代码块。结构：
{
  "location_id": "场景编号",
  "land_type": "上述7个英文名称之一",
  "reason": "简短判断依据，30至120字",
  "confidence": 0.0,
  "alternatives": ["次可能类别名，没有则为空数组"]
}
""".strip()


def build_prediction_prompt(location_id: str, graph_text: str) -> str:
    return f"""
场景编号：{location_id}

以下是该地点的场景图（纯文本，不含图片）：
{graph_text}

请只依据上述场景图推断该地点的主导用地类型，从7个类别中选择最符合的1类。
只输出一个合法JSON对象。
""".strip()


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("landtype_prediction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / "build_landtype_prediction.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def scene_sort_key(path: Path):
    m = re.fullmatch(r"scene_(\d+)", path.stem)
    return (0, int(m.group(1))) if m else (1, path.stem)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def serialize_scene_graph(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes", []) or []
    relations = graph.get("relations", []) or []
    id_to_node = {n.get("id"): n for n in nodes if isinstance(n, dict)}

    def clean(v: Any) -> str:
        t = str(v if v is not None else "").strip()
        return "" if t.lower() == "none" else t

    lines = ["[Scene Nodes]"]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        parts = [f"Node {node.get('id')}", f"Type={clean(node.get('type')) or 'unknown'}"]
        if clean(node.get("function")):
            parts.append(f"Function={clean(node.get('function'))}")
        if clean(node.get("feature")):
            parts.append(f"Feature={clean(node.get('feature'))}")
        if clean(node.get("text")):
            parts.append(f"Sign/Text={clean(node.get('text'))}")
        lines.append("- " + "; ".join(parts))

    lines.append("")
    lines.append("[Inter-node Relations]")

    def node_tag(nid: Any) -> str:
        n = id_to_node.get(nid)
        if n is None:
            return f"Node {nid}"
        t = clean(n.get("type"))
        fn = clean(n.get("function"))
        suffix = f"{t}/{fn}" if fn else t
        return f"Node {nid} ({suffix or 'unknown'})"

    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rp = []
        if clean(rel.get("Spatial_relation")):
            rp.append(f"spatial: {clean(rel.get('Spatial_relation'))}")
        if clean(rel.get("Semantic_relation")):
            rp.append(f"semantic: {clean(rel.get('Semantic_relation'))}")
        if clean(rel.get("Topological_relation")):
            rp.append(f"topo: {clean(rel.get('Topological_relation'))}")
        lines.append(f"- {node_tag(rel.get('from'))} -- [{', '.join(rp)}] -> {node_tag(rel.get('to'))}")
    return "\n".join(lines)


def invoke_text(client, system_prompt: str, user_prompt: str, temperature: float) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        max_tokens=1024, temperature=temperature, top_p=0.7,
    )
    return resp.choices[0].message.content


def parse_json_response(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty model response")
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
            raise ValueError("No valid JSON object found in model output")
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def normalize_land_type(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"land_type is not a string: {value!r}")
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


def validate_prediction(data: dict[str, Any], location_id: str) -> dict[str, Any]:
    land_type = normalize_land_type(data.get("land_type"))
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise ValueError("Missing reason")
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence is not numeric") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    alternatives = []
    for item in data.get("alternatives", []) or []:
        try:
            alt = normalize_land_type(item)
        except ValueError:
            continue
        if alt != land_type and alt not in alternatives:
            alternatives.append(alt)

    return {
        "location_id": location_id,
        "land_type": land_type,
        "reason": reason,
        "confidence": round(confidence, 4),
        "alternatives": alternatives[:2],
    }


def is_valid_prediction(path: Path) -> bool:
    data = load_json(path)
    return bool(data and data.get("status") == "complete" and data.get("land_type") in LAND_TYPES)


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def process_scene(client, sg_path: Path, out_path: Path, logger, max_retries: int) -> None:
    loc_id = sg_path.stem
    graph = load_json(sg_path)
    if graph is None:
        raise ValueError(f"Cannot parse scene graph: {sg_path.name}")
    graph_text = serialize_scene_graph(graph)
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_prediction_prompt(loc_id, graph_text)
            if last_error:
                prompt += f"\n\nPrevious validation failed: {last_error}. Please fix and re-output JSON."
            raw = invoke_text(client, SYSTEM_PROMPT, prompt, temperature=0.1)
            result = validate_prediction(parse_json_response(raw), loc_id)
            result["num_nodes"] = len(graph.get("nodes", []) or [])
            result["num_relations"] = len(graph.get("relations", []) or [])
            result["prediction_source"] = "scene_graph"
            result["model"] = MODEL_NAME
            result["status"] = "complete"
            write_json_atomic(out_path, result)
            logger.info("%s -> %s (conf=%.2f)", loc_id, result["land_type"], result["confidence"])
            return
        except Exception as exc:
            last_error = str(exc)
            logger.warning("%s attempt %d/%d failed: %s", loc_id, attempt, max_retries, last_error)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Max retries exceeded: {last_error}")


def evaluate(scene_graph_dir: Path, gt_dir: Path, pred_dir: Path, logger=None) -> dict:
    scene_paths = sorted(scene_graph_dir.glob("scene_*.json"), key=scene_sort_key)
    records = []
    missing_pred, missing_gt = [], []

    for sp in scene_paths:
        sid = sp.stem
        gt = load_json(gt_dir / f"{sid}.json")
        pred = load_json(pred_dir / f"{sid}.json")
        true_label = gt.get("land_type") if gt else None
        pred_label = pred.get("land_type") if pred else None
        if true_label not in LAND_TYPES:
            missing_gt.append(sid)
            continue
        if pred_label not in LAND_TYPES:
            missing_pred.append(sid)
            continue
        records.append({
            "scene_id": sid, "true": true_label, "pred": pred_label,
            "correct": true_label == pred_label,
            "confidence": pred.get("confidence", ""),
            "num_nodes": pred.get("num_nodes", ""),
            "reason": str(pred.get("reason", "")).replace("\n", " ").strip(),
        })

    if not records:
        raise RuntimeError("No evaluable samples (need both predictions and ground truth)")

    n = len(records)
    correct = sum(r["correct"] for r in records)
    accuracy = correct / n
    support = Counter(r["true"] for r in records)
    pred_count = Counter(r["pred"] for r in records)
    present = [l for l in LAND_TYPES if support[l] > 0]
    majority_base = max(support.values()) / n

    confusion = {t: {p: 0 for p in LAND_TYPES} for t in LAND_TYPES}
    for r in records:
        confusion[r["true"]][r["pred"]] += 1

    per_class = {}
    for label in LAND_TYPES:
        tp = confusion[label][label]
        fp = pred_count[label] - tp
        fn = support[label] - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[label] = {"support": support[label], "predicted": pred_count[label],
                            "tp": tp, "fp": fp, "fn": fn,
                            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    macro_p = sum(per_class[c]["precision"] for c in present) / len(present)
    macro_r = sum(per_class[c]["recall"] for c in present) / len(present)
    macro_f1 = sum(per_class[c]["f1"] for c in present) / len(present)
    weighted_f1 = sum(support[c] / n * per_class[c]["f1"] for c in present)

    report = {
        "model": MODEL_NAME, "num_evaluated": n, "num_correct": correct,
        "accuracy": round(accuracy, 4), "majority_baseline": round(majority_base, 4),
        "macro_precision": round(macro_p, 4), "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
        "present_classes": present, "per_class": per_class,
        "confusion_matrix": confusion,
    }
    write_json_atomic(pred_dir / "evaluation_report.json", report)

    with (pred_dir / "prediction_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "true", "pred", "correct", "confidence", "num_nodes", "reason"])
        for r in sorted(records, key=lambda x: int(re.search(r"\d+", x["scene_id"]).group()) if re.search(r"\d+", x["scene_id"]) else 0):
            w.writerow([r["scene_id"], r["true"], r["pred"], int(r["correct"]),
                         r["confidence"], r["num_nodes"], r["reason"]])

    with (pred_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["land_type", "support", "predicted", "TP", "FP", "FN", "precision", "recall", "f1"])
        for label in LAND_TYPES:
            m = per_class[label]
            w.writerow([label, m["support"], m["predicted"], m["tp"], m["fp"], m["fn"],
                         m["precision"], m["recall"], m["f1"]])

    with (pred_dir / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [SHORT_LABEL[p] for p in LAND_TYPES])
        for t in LAND_TYPES:
            w.writerow([SHORT_LABEL[t]] + [confusion[t][p] for p in LAND_TYPES])

    print(f"\n===== Land-Use Prediction Evaluation =====")
    print(f"Evaluated scenes: {n}; correct: {correct}; wrong: {n - correct}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{n})")
    print(f"Majority baseline: {majority_base:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    return report


def main() -> None:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOG_DIR)

    if ZhipuAiClient is None:
        raise RuntimeError("Please install the zai package: pip install zai")
    if API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("Please set ZHIPU_API_KEY environment variable")
    if not SCENEGRAPH_DIR.is_dir():
        raise FileNotFoundError(f"Scene graph directory not found: {SCENEGRAPH_DIR}")
    if not GROUND_TRUTH_DIR.is_dir():
        raise FileNotFoundError(f"Ground truth directory not found: {GROUND_TRUTH_DIR}")

    client = ZhipuAiClient(api_key=API_KEY)
    scene_paths = sorted(SCENEGRAPH_DIR.glob("scene_*.json"), key=scene_sort_key)
    if LIMIT > 0:
        scene_paths = scene_paths[:LIMIT]

    pending = [sp for sp in scene_paths if not is_valid_prediction(PREDICTION_DIR / f"{sp.stem}.json")]
    logger.info("Total: %d, to predict: %d, already done: %d",
                len(scene_paths), len(pending), len(scene_paths) - len(pending))

    failed = []
    for i, sp in enumerate(pending, 1):
        out = PREDICTION_DIR / f"{sp.stem}.json"
        logger.info("[%d/%d] Predicting %s", i, len(pending), sp.stem)
        try:
            process_scene(client, sp, out, logger, MAX_RETRIES)
        except Exception as exc:
            failed.append({"scene": sp.stem, "error": str(exc)})
            logger.error("%s failed: %s", sp.stem, exc)

    write_json_atomic(LOG_DIR / "failed_landtype.json", {"failed": failed})
    logger.info("Prediction done: %d success, %d failed", len(pending) - len(failed), len(failed))
    evaluate(SCENEGRAPH_DIR, GROUND_TRUTH_DIR, PREDICTION_DIR, logger)


if __name__ == "__main__":
    main()
