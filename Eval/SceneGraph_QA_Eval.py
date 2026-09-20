#!/usr/bin/env python3
"""
Scene-graph-based street-view QA evaluation.

Loads QA samples from a directory, retrieves the corresponding scene graph
JSON, sends the structured scene graph + question to a VLM, and computes
per-category and overall accuracy.

Usage:
    python SceneGraph_QA_Eval.py \
        --qa-dir StreetQADataset_v1 \
        --sg-dir Outputs/GLM/BSG_QA_v1 \
        --output-dir QA_experiments/BSG_QA_GLM \
        --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _import_zai_client() -> Any:
    try:
        from zai import ZhipuAiClient
        return ZhipuAiClient
    except ImportError:
        raise ImportError(
            "The 'zai' package is required. Install it with: pip install zai"
        )


ZhipuAiClient = _import_zai_client()

MODEL_NAME = "GLM-4-Flash"
QUESTION_CATEGORIES = ("EC", "AD", "SD", "Text", "CS", "RC")

SYSTEM_PROMPT = (
    "You are a scene-graph QA assistant.\n"
    "You will be given a structured description of a scene graph and a multiple-choice question.\n"
    "Answer strictly based on the scene graph information only; do not add external knowledge.\n"
    "Return only one option label (e.g. A, B, C, D) or the full option text. No explanation."
)


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("，", ",").replace("。", ".").replace("：", ":")
    text = text.replace("；", ";").replace("？", "?").replace("！", "!")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \n\t\r\f\v。,.?!:;")
    return text.lower()


def safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_qa_files(qa_dir: Path) -> list[tuple[str, list[dict[str, Any]], Path]]:
    samples = []
    for path in sorted(qa_dir.glob("*.json")):
        data = safe_load_json(path)
        if not data:
            continue
        sample_id = str(data.get("sample_id", path.stem))
        questions = data.get("questions")
        if not isinstance(questions, list):
            continue
        samples.append((sample_id, questions, path))
    return samples


def build_scene_graph_text(scene_graph: dict[str, Any]) -> str:
    nodes = scene_graph.get("nodes", [])
    relations = scene_graph.get("relations", [])
    lines = ["Scene graph nodes:"]
    for node in nodes:
        nid = node.get("id")
        ntype = str(node.get("type", "")).strip()
        feature = str(node.get("feature", "")).strip()
        text = str(node.get("text", "")).strip()
        summary = ntype
        if feature:
            summary += f", {feature}"
        if text:
            summary += f", text: {text}"
        lines.append(f"- Node {nid}: {summary}")
    lines.append("Scene graph relations:")
    for rel in relations:
        parts = []
        spatial = str(rel.get("Spatial_relation", "")).strip()
        semantic = str(rel.get("Semantic_relation", "")).strip()
        topo = str(rel.get("Topological_relation", "")).strip()
        if spatial:
            parts.append(spatial)
        if semantic:
            parts.append(semantic)
        if topo:
            parts.append(topo)
        lines.append(f"- Node {rel.get('from')} -> Node {rel.get('to')}: {', '.join(parts)}")
    return "\n".join(lines)


def build_prompt(sg_text: str, question: dict[str, Any]) -> str:
    qtext = str(question.get("question", "")).strip()
    options_text = []
    for opt in question.get("options", []):
        label = str(opt.get("label", "")).strip()
        otext = str(opt.get("text", "")).strip()
        if label and otext:
            options_text.append(f"{label}. {otext}")
    return (
        "Answer the question based on the scene graph description below.\n\n"
        f"{sg_text}\n\n"
        f"Question: {qtext}\n"
        "Options:\n" + "\n".join(options_text) +
        "\n\nReturn only an option label (A/B/C/D) or the full option text."
    )


def extract_text_from_response(raw: Any) -> str:
    if isinstance(raw, list):
        texts = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts).strip()
    return str(raw).strip()


def select_answer_label(raw_answer: str, options: list[dict[str, Any]]) -> str | None:
    normalized = normalize_text(raw_answer)
    if not normalized:
        return None
    m = re.search(r"\b([a-dA-D])\b", normalized)
    if m and any(o.get("label") == m.group(1).upper() for o in options):
        return m.group(1).upper()
    for opt in options:
        if normalize_text(opt.get("text", "")) == normalized:
            return opt.get("label")
    for opt in options:
        if str(opt.get("label", "")).upper() in normalized:
            return str(opt.get("label", "")).upper()
    for opt in options:
        if normalize_text(opt.get("text", "")) in normalized:
            return opt.get("label")
    return None


def ask_question(client, question, sg_text, model_name, temperature=0.0, max_retries=5, base_delay=0.8):
    prompt = build_prompt(sg_text, question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages,
                max_tokens=256, temperature=temperature, top_p=0.8,
            )
            content = resp.choices[0].message.content
            answer_text = extract_text_from_response(content)
            answer_label = select_answer_label(answer_text, question.get("options", []))
            return answer_label or "", answer_text, json.dumps(content, ensure_ascii=False)
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            logging.getLogger(__name__).warning(
                "Request failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt, max_retries, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"Model request failed after {max_retries} retries: {last_exc}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_evaluation(qa_dir: Path, sg_dir: Path, output_dir: Path,
                   api_key: str, model_name: str, limit=None,
                   max_retries=5, retry_base_delay=0.8) -> None:
    client = ZhipuAiClient(api_key=api_key)
    samples = load_qa_files(qa_dir)
    if not samples:
        raise RuntimeError(f"No valid QA files found in {qa_dir}")

    output_dir = output_dir.resolve()
    ensure_dir(output_dir)
    results_path = output_dir / "scene_graph_qa_results.csv"
    detail_path = output_dir / "scene_graph_qa_details.json"

    eval_results = []
    stats = {c: {"total": 0, "correct": 0} for c in QUESTION_CATEGORIES}
    stats["overall"] = {"total": 0, "correct": 0}
    missing_sg = 0
    processed = 0

    for sample_id, questions, qa_path in samples:
        if limit is not None and processed >= limit:
            break
        sg_path = sg_dir / f"{sample_id}.json"
        if not sg_path.exists():
            missing_sg += 1
            continue
        sg = safe_load_json(sg_path)
        if not sg:
            missing_sg += 1
            continue

        sg_text = build_scene_graph_text(sg)
        for q in questions:
            cat = str(q.get("category", "Unknown"))
            stats.setdefault(cat, {"total": 0, "correct": 0})
            stats[cat]["total"] += 1
            stats["overall"]["total"] += 1

            gold = str(q.get("answer_label", "")).strip().upper()
            pred_label, pred_text, raw = ask_question(
                client, q, sg_text, model_name=model_name,
                temperature=0.0, max_retries=max_retries, base_delay=retry_base_delay,
            )
            correct = gold != "" and pred_label == gold
            if correct:
                stats[cat]["correct"] += 1
                stats["overall"]["correct"] += 1

            eval_results.append({
                "sample_id": sample_id,
                "qa_file": str(qa_path),
                "scene_graph_file": str(sg_path),
                "category": cat,
                "question": q.get("question", ""),
                "gold_label": gold,
                "gold_text": q.get("answer_text", ""),
                "pred_label": pred_label,
                "pred_text": pred_text,
                "raw_response": raw,
                "correct": correct,
                "model": model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            })
        processed += 1

    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_results[0].keys()) if eval_results else
            ["sample_id", "qa_file", "scene_graph_file", "category", "question",
             "gold_label", "gold_text", "pred_label", "pred_text",
             "raw_response", "correct", "model", "timestamp"])
        writer.writeheader()
        writer.writerows(eval_results)

    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print(f"\n===== Scene Graph QA Evaluation =====")
    print(f"Processed samples: {processed}")
    print(f"Missing scene graphs: {missing_sg}")
    print(f"Total questions: {stats['overall']['total']}")
    print(f"Overall correct: {stats['overall']['correct']}")
    acc = stats["overall"]["correct"] / stats["overall"]["total"] if stats["overall"]["total"] else 0.0
    print(f"Overall accuracy: {acc:.4f}\n")
    for cat in QUESTION_CATEGORIES:
        t = stats[cat]["total"]
        c = stats[cat]["correct"]
        a = c / t if t else 0.0
        print(f"{cat}: total={t}, correct={c}, accuracy={a:.4f}")
    print(f"\nResults saved to: {results_path}")
    print(f"Details saved to: {detail_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Scene-graph-based street-view QA evaluation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--qa-dir", type=Path, default=Path("StreetQADataset_v1"),
                   help="Directory with ground-truth QA JSON files")
    p.add_argument("--sg-dir", type=Path, default=Path("Outputs/GLM/BSG_QA_v1"),
                   help="Directory with predicted scene graph JSON files")
    p.add_argument("--output-dir", type=Path, default=Path("QA_experiments/BSG_QA_GLM"),
                   help="Output directory for evaluation results")
    p.add_argument("--api-key", type=str, default=os.environ.get("ZHIPU_API_KEY", ""),
                   help="Zhipu API key (or set ZHIPU_API_KEY env var)")
    p.add_argument("--model-name", type=str, default=MODEL_NAME, help="VLM model name")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    p.add_argument("--retry-base-delay", type=float, default=0.8, help="Base retry delay (seconds)")
    p.add_argument("--limit", type=int, default=None, help="Process only first N samples (debug)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.api_key.strip():
        print("Error: no API key provided. Use --api-key or set ZHIPU_API_KEY.", file=sys.stderr)
        return 1
    if not args.qa_dir.is_dir():
        print(f"Error: QA directory not found: {args.qa_dir}", file=sys.stderr)
        return 1
    if not args.sg_dir.is_dir():
        print(f"Error: scene graph directory not found: {args.sg_dir}", file=sys.stderr)
        return 1

    try:
        run_evaluation(
            qa_dir=args.qa_dir, sg_dir=args.sg_dir, output_dir=args.output_dir,
            api_key=args.api_key, model_name=args.model_name, limit=args.limit,
            max_retries=args.max_retries, retry_base_delay=args.retry_base_delay,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
