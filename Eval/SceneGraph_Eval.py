"""
Scene graph evaluation metrics.

Computes entity accuracy, structure accuracy, triplet accuracy
(spatial / topological / overall), and weighted recall over a
directory of ground-truth and predicted scene-graph JSON files.

Usage:
    python SceneGraph_Eval.py
"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Entity and relation vocabulary (Chinese labels)
# ---------------------------------------------------------------------------
ENTITY_ID_DICT = {
    "建筑物": 1, "围墙": 2, "围栏": 3, "围挡": 4, "桥梁": 5,
    "人行天桥": 6, "高架桥": 7, "台阶": 8, "柱子": 9, "墙体": 10,
    "支撑结构": 11, "阳台": 12, "门": 13, "墙面": 14, "道路": 15,
    "人行道": 16, "斑马线": 17, "路缘石": 18, "路面": 19, "停车场": 20,
    "停车区域": 21, "交通信号灯": 22, "交通标志": 23, "路牌": 24, "道路标线": 25,
    "减速带": 26, "隔离栏": 27, "石墩": 28, "隔离桩": 29, "隔离带": 30,
    "交通锥": 31, "路障": 32, "树木": 33, "灌木丛": 34, "花卉": 35,
    "草地": 36, "花坛": 37, "含花箱": 38, "花盆": 39, "绿化带": 40,
    "含绿化岛": 41, "植被": 42, "植物": 43, "路灯": 44, "路灯杆": 45,
    "电线杆": 46, "监控杆": 47, "标识杆": 48, "公交站": 49, "站牌": 50,
    "垃圾桶": 51, "垃圾箱": 52, "消防栓": 53, "井盖": 54, "配电箱": 55,
    "设备箱": 56, "监控设备": 57, "商铺": 58, "商店": 59, "招牌": 60,
    "广告牌": 61, "标识牌": 62, "宣传栏": 63, "信息牌": 64, "装载机": 65,
    "起重机": 66, "塔吊": 67, "施工设备": 68, "雕塑": 69, "长椅": 70,
    "座椅": 71, "桌子": 72, "遮阳棚": 73, "帐篷": 74, "横幅": 75,
    "管道": 76, "空调": 77, "涂鸦": 78, "国旗": 79, "太阳能板": 80,
    "通信塔": 81, "手推车": 82, "防护网": 83, "广场": 84, "集装箱": 85,
    "储罐": 86, "货柜": 87, "施工围挡": 88,
}

SPATIAL_MAP = {
    "左侧": 1, "左边": 1, "左": 1, "左方": 1, "左部": 1, "左旁": 1, "左外侧": 1, "靠左": 1,
    "右侧": 2, "右边": 2, "右": 2, "右方": 2, "右部": 2, "右旁": 2, "右外侧": 2, "靠右": 2,
    "前方": 3, "前边": 3, "前": 3, "前端": 3, "前部": 3, "前方远处": 3, "正前方": 3,
    "后方": 4, "后边": 4, "后": 4, "后端": 4, "后部": 4, "正后方": 4,
    "上方": 5, "上面": 5, "上": 5, "上部": 5, "顶部": 5, "顶上": 5, "正上方": 5,
    "下方": 6, "下面": 6, "下": 6, "下部": 6, "底部": 6, "底下": 6, "正下方": 6,
    "左上方": 7, "左上": 7,
    "右上方": 8, "右上": 8,
    "左下方": 9, "左下": 9,
    "右下方": 10, "右下": 10,
    "左前": 11, "左前方": 11,
    "右前": 12, "右前方": 12,
    "左后": 13, "左后方": 13,
    "右后": 14, "右后方": 14,
    "中央": 15, "中间": 15, "正中": 15, "中心": 15,
    "边缘": 16, "边沿": 16, "边上": 16,
    "正面": 17, "正脸": 17,
    "侧面": 18, "侧边": 18, "侧方": 18,
    "表面": 19, "面上": 19, "表层": 19,
    "路边": 20, "路旁": 20, "道路旁": 20, "路边上": 20,
    "远处": 21, "远方": 21, "较远": 21,
    "近处": 22, "近旁": 22, "附近": 22, "较近": 22,
    "周围": 23, "四周": 23, "周边": 23,
    "两侧": 24, "两边": 24, "两旁": 24, "左右两侧": 24,
    "对侧": 25, "对面": 25, "相对侧": 25,
    "相邻侧": 26, "相邻": 26, "旁边": 26,
    "上方右侧": 8, "下方右侧": 10, "上方左侧": 7, "下方左侧": 9,
}

TOPO_MAP = {
    "相邻": 1, "邻接": 1, "接触": 1, "相接": 1, "相连": 1, "紧靠": 1, "贴近": 1, "挨着": 1,
    "环绕": 2, "包围": 2, "围绕": 2, "包绕": 2, "一圈": 2,
    "属于": 3, "附属": 3, "依附": 3, "附属于": 3,
    "包含": 4, "内部": 4, "里面": 4, "在内": 4,
    "被包含": 5, "在内部": 5, "在里面": 5,
    "嵌入": 6, "嵌套": 6, "插入": 6, "镶入": 6,
    "底层": 7, "下层": 7, "底部层": 7,
    "上层": 8, "顶部层": 8,
    "下层": 9, "中间层": 9,
    "连接": 10, "连通": 10, "衔接": 10,
    "附着": 11, "安装于": 11, "固定于": 11, "悬挂于": 11, "贴于": 11, "挂于": 11, "装在": 11,
    "支撑": 12, "托起": 12, "承托": 12, "撑住": 12,
    "放置于": 13, "放在": 13, "置于": 13,
    "种植于": 14, "长在": 14, "栽于": 14,
    "延伸": 15, "延展": 15, "伸展": 15,
    "平行": 16, "并行": 16, "相平": 16,
    "覆盖": 17, "盖住": 17, "遮盖": 17,
    "被覆盖": 18, "被盖住": 18,
    "遮挡": 19, "挡住": 19, "遮蔽": 19,
    "被遮挡": 20, "被挡住": 20,
    "分隔": 21, "隔开": 21, "分开": 21, "隔离": 21,
    "穿越": 22, "跨越": 22, "穿过": 22, "跨过": 22,
    "背靠": 23, "背向": 23, "背对": 23,
}

ID2ENTITY = {v: k for k, v in ENTITY_ID_DICT.items()}
ID2SPATIAL = {}
for k, v in SPATIAL_MAP.items():
    ID2SPATIAL.setdefault(v, k)
ID2TOPO = {}
for k, v in TOPO_MAP.items():
    ID2TOPO.setdefault(v, k)


class SceneGraphEvaluator:
    """Evaluate predicted scene graphs against ground truth."""

    def __init__(self):
        self.entity_gt_total = defaultdict(int)
        self.entity_pred_correct = defaultdict(int)
        self.spatial_gt_total = defaultdict(int)
        self.spatial_pred_correct = defaultdict(int)
        self.topo_gt_total = defaultdict(int)
        self.topo_pred_correct = defaultdict(int)
        self.sample_metrics = []

    @staticmethod
    def load_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def evaluate_entity(self, gt_nodes, pred_nodes):
        gt_eids = set()
        gt_type_count = defaultdict(int)
        for n in gt_nodes:
            eid = ENTITY_ID_DICT.get(n.get("type", ""), -1)
            if eid != -1:
                gt_eids.add(eid)
                gt_type_count[eid] += 1
        for eid, cnt in gt_type_count.items():
            self.entity_gt_total[eid] += cnt

        pred_eids = set()
        for n in pred_nodes:
            eid = ENTITY_ID_DICT.get(n.get("type", ""), -1)
            if eid != -1:
                pred_eids.add(eid)

        correct_eids = gt_eids & pred_eids
        for eid in correct_eids:
            self.entity_pred_correct[eid] += gt_type_count.get(eid, 0)

        total = len(gt_eids) if gt_eids else 1
        return round(len(correct_eids) / total, 4)

    def evaluate_triplet(self, gt_nodes, pred_nodes, gt_rels, pred_rels):
        def build_node_map(nodes):
            m = {}
            for n in nodes:
                nid = n.get("id")
                eid = ENTITY_ID_DICT.get(n.get("type", ""), -1)
                if nid is not None and eid != -1:
                    m[nid] = eid
            return m

        gt_nmap = build_node_map(gt_nodes)
        pred_nmap = build_node_map(pred_nodes)

        def get_triplets(rels, node_map, is_spatial):
            struct_set, triplet_set = set(), set()
            rel_count = defaultdict(int)
            for r in rels:
                f, t = r.get("from"), r.get("to")
                if f not in node_map or t not in node_map:
                    continue
                rel_str = r.get("Spatial_relation" if is_spatial else "Topological_relation", "")
                rid = SPATIAL_MAP.get(rel_str, -1) if is_spatial else TOPO_MAP.get(rel_str, -1)
                if rid == -1:
                    continue
                struct_set.add((node_map[f], node_map[t]))
                triplet_set.add((node_map[f], node_map[t], rid))
                rel_count[rid] += 1
            return struct_set, triplet_set, rel_count

        gt_sp_struct, gt_sp_trip, gt_sp_cnt = get_triplets(gt_rels, gt_nmap, True)
        pred_sp_struct, pred_sp_trip, _ = get_triplets(pred_rels, pred_nmap, True)
        gt_tp_struct, gt_tp_trip, gt_tp_cnt = get_triplets(gt_rels, gt_nmap, False)
        pred_tp_struct, pred_tp_trip, _ = get_triplets(pred_rels, pred_nmap, False)

        for rid, cnt in gt_sp_cnt.items():
            self.spatial_gt_total[rid] += cnt
        for rid, cnt in gt_tp_cnt.items():
            self.topo_gt_total[rid] += cnt

        correct_sp = gt_sp_trip & pred_sp_trip
        for _ in correct_sp:
            self.spatial_pred_correct[_[2]] += 1
        correct_tp = gt_tp_trip & pred_tp_trip
        for _ in correct_tp:
            self.topo_pred_correct[_[2]] += 1

        sp_acc = len(correct_sp) / len(gt_sp_trip) if gt_sp_trip else 0.0
        tp_acc = len(correct_tp) / len(gt_tp_trip) if gt_tp_trip else 0.0

        gt_struct = gt_sp_struct | gt_tp_struct
        pred_struct = pred_sp_struct | pred_tp_struct
        gt_trip = gt_sp_trip | gt_tp_trip
        pred_trip = pred_sp_trip | pred_tp_trip

        struct_acc = len(gt_struct & pred_struct) / len(gt_struct) if gt_struct else 0.0
        all_trip_acc = len(gt_trip & pred_trip) / len(gt_trip) if gt_trip else 0.0

        return (
            round(struct_acc, 4),
            round(all_trip_acc, 4),
            round(sp_acc, 4),
            round(tp_acc, 4),
            round(all_trip_acc, 4),
        )

    def eval_one(self, gt, pred):
        if not gt or not pred:
            return None
        try:
            ent_acc = self.evaluate_entity(gt["nodes"], pred["nodes"])
            struct_acc, all_trip, sp_trip, tp_trip, rel_acc = self.evaluate_triplet(
                gt["nodes"], pred["nodes"], gt["relations"], pred["relations"]
            )
            overall = np.mean([ent_acc, struct_acc, all_trip])
            res = {
                "entity_acc": ent_acc,
                "structure_acc": struct_acc,
                "all_triplet_acc": all_trip,
                "spatial_triplet_acc": sp_trip,
                "topo_triplet_acc": tp_trip,
                "relation_acc": rel_acc,
                "overall_acc": round(overall, 4),
            }
            self.sample_metrics.append(res)
            return res
        except Exception:
            return None

    def compute_weighted_recall(self):
        ent_gt = sum(self.entity_gt_total.values())
        ent_corr = sum(self.entity_pred_correct.values())
        sp_gt = sum(self.spatial_gt_total.values())
        sp_corr = sum(self.spatial_pred_correct.values())
        tp_gt = sum(self.topo_gt_total.values())
        tp_corr = sum(self.topo_pred_correct.values())
        return {
            "entity_weighted_recall": round(ent_corr / ent_gt, 4) if ent_gt else 0.0,
            "spatial_weighted_recall": round(sp_corr / sp_gt, 4) if sp_gt else 0.0,
            "topo_weighted_recall": round(tp_corr / tp_gt, 4) if tp_gt else 0.0,
            "relation_weighted_recall": round((sp_corr + tp_corr) / (sp_gt + tp_gt), 4)
            if (sp_gt + tp_gt) else 0.0,
        }

    def run(self, gt_dir, pred_dir, csv_name="result.csv"):
        filenames = [f for f in os.listdir(gt_dir) if f.endswith(".json")]
        results = []
        for fname in tqdm(filenames, desc="Evaluating files"):
            pred_path = os.path.join(pred_dir, fname)
            if not os.path.exists(pred_path):
                continue
            gt_data = self.load_json(os.path.join(gt_dir, fname))
            pred_data = self.load_json(pred_path)
            r = self.eval_one(gt_data, pred_data)
            if r:
                r["filename"] = fname
                results.append(r)

        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(csv_name, index=False, encoding="utf-8-sig")

        wr = self.compute_weighted_recall()
        print(f"\n===== Scene Graph Evaluation ({len(df)} samples) =====")
        if not df.empty:
            print(f"Entity accuracy:      {df['entity_acc'].mean():.3f}")
            print(f"Structure accuracy:  {df['structure_acc'].mean():.3f}")
            print(f"Overall triplet acc: {df['all_triplet_acc'].mean():.3f}")
            print(f"Spatial triplet acc: {df['spatial_triplet_acc'].mean():.3f}")
            print(f"Topo triplet acc:    {df['topo_triplet_acc'].mean():.3f}")
        print(f"\n--- Weighted Recall ---")
        print(f"Entity:    {wr['entity_weighted_recall']:.4f}")
        print(f"Spatial:   {wr['spatial_weighted_recall']:.4f}")
        print(f"Topology:  {wr['topo_weighted_recall']:.4f}")
        print(f"Relation:  {wr['relation_weighted_recall']:.4f}")
        print("=" * 50)
        return df


if __name__ == "__main__":
    evaluator = SceneGraphEvaluator()
    evaluator.run(
        gt_dir="../Datasets/SingleView_constrain_classes_15",
        pred_dir="../Datasets/SingleView_constrain_classes",
        csv_name="result.csv",
    )
