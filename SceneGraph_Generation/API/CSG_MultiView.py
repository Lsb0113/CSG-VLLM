"""
Multi-view scene graph generation via Zhipu GLM-4.6V-Flash API.

For each location with four panoramic street-view images (0/90/180/270
degrees), this script merges all four views into a single unified semantic
scene graph (nodes + relations) and stores it as one JSON file per location.

Usage:
    python CSG_MultiView.py

Before running, set your API key:
    export ZHIPU_API_KEY="your-api-key-here"
"""

import base64
import io
import json
import os

from PIL import Image
from zai import ZhipuAiClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "GLM-4.6V-Flash"
IMAGES_DIR = "Datasets/MultiViewDataset_new"   # one subdirectory per location
OUTPUT_DIR = "Outputs/GLM/MultiViews_new"
LOG_DIR = "logs"

# File naming convention inside each location directory. The four images are
# opened in the original notebook order [front(90), back(270), left(0),
# right(180)]. NOTE: the prompt below describes the four input slots as
# left/front/right/back; verify this ordering against your own data before use.
VIEW_FILES = ("_90_0.jpg", "_270_0.jpg", "_0_0.jpg", "_180_0.jpg")


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def invoke_with_image(client: ZhipuAiClient, query: str, images) -> str:
    """Send up to four images plus the text query to the VLM."""
    images = list(images)
    base64_images = [encode_image_to_base64(img) for img in images]

    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
    for img_b64 in base64_images:
        messages[0]["content"].append(
            {"type": "image_url", "image_url": {"url": img_b64}}
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=8192,
        temperature=0.5,
        top_p=0.7,
    )
    return response.choices[0].message.content


def log_error(msg: str, error_file) -> None:
    """Print an error to the console and append it to the log file."""
    print(msg)
    print(msg, file=error_file)
    error_file.flush()


def need_process(output_dir: str, img_name: str) -> bool:
    """Return True if the scene graph for this location has not been generated yet."""
    json_path = os.path.join(output_dir, img_name + ".json")
    if not os.path.exists(json_path):
        return True
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return not data  # empty output -> reprocess


def extract_json_format(data_str: str) -> str:
    """Extract the JSON object from the model output (strip any surrounding text)."""
    json_start = data_str.find("{")
    json_end = data_str.rfind("}") + 1
    return data_str[json_start:json_end]


# ---------------------------------------------------------------------------
# Entity / relation vocabularies (Chinese labels, as output by the GLM model)
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

# Spatial relation vocabulary (Chinese synonyms -> canonical ID)
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

# Topological relation vocabulary (Chinese synonyms -> canonical ID)
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


# Prompt sent to the VLM. Kept in Chinese: the model is prompted in Chinese so
# that its output vocabulary matches the entity / relation dictionaries above.
QUERY = """
你是专业的二维语义场景图构建专家。
我将给你**4张按顺序排列的全景图：第1张=左视角，第2张=前视角，第3张=右视角，第4张=后视角**。
请把这4张图**融合成一个完整、统一、全局的二维语义场景图**，而不是分别生成4张图的场景图。

### 一、核心结构要求
1.  严格遵循「节点(nodes) + 关系(relations)」的JSON格式，不得出现格式错误、字段缺失
2.  节点(nodes)要求：
    - 每个节点必须包含：id：从1开始递增唯一整数。 type：必须高度概括，包括但不限于如：建筑物、道路、树木、店铺、围栏、标识、路牌、路灯、交通灯。 feature：必须包含外观特征：颜色、形状、材质、结构细节（阳台、空调机、外墙、玻璃等）。 function：实体功能，简洁准确。text：提取实体表面可见文字（路名、招牌），无则为空字符串。
    - **强制要求：绝对不提取「行人」「车辆」「非机动车」「骑手」这类动态、易变化的实体，仅提取固定不变的静态场景元素**
    - 必须完整识别画面中所有核心静态实体：道路、建筑物、交通设施、绿化树木、围栏围墙、交通标识、路灯、商铺等，不得遗漏关键静态元素
    - 不要有冗余信息，比如：实体外观有文字信息，feature里不保存文字信息，而是保存在text中
3.  关系(relations)要求：
    - 每个关系必须包含：from：起点节点id、to：终点节点id、"Spatial_relation": "空间位置（包括但不限于如：左侧、右侧、前方、后方、上方、下方、旁边、路边、远处、相邻。）", "Semantic_relation": "功能语义（包括但不限于如：指挥、引导、指示、服务、防护、装饰、照明、分隔、交通控制。）", "Topological_relation": "拓扑关系（包括但不限于如：属于、包含、附属、底层、上层、连接、相邻、嵌套。）"
    - 空间关系严格符合画面
    - 功能关系符合现实逻辑
    - 仅保留静态实体之间的关联关系，确保场景逻辑闭环

### 二、输出要求
1.  严格控制输出长度：仅提取核心、必要的静态实体，避免过度细分冗余实体，确保JSON完整输出，无截断
2.  优先保证格式完整：必须保证JSON结构完整、语法正确，可直接解析
3.  禁止任何额外内容：仅输出标准JSON，不得添加任何说明、注释、解释性文字，不得截断JSON结构

### 三、质量要求
1.  合并同一实体：不同图片可能存在同一个实体，可能因为视角原因显示不全，但是需要推理出哪些内容属于同一实体，减少同一实体信息冗余。
2.  100%贴合画面：所有静态实体、特征、位置关系完全对应图片内容，无捏造
3.  语义精准：实体分类、关系描述符合学术规范，可直接用于模型输入
4.  结构完整：无遗漏核心静态实体、无缺失关键关系、无格式错误
"""


def main() -> None:
    if API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Error: please set the ZHIPU_API_KEY environment variable.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    error_file = open(os.path.join(LOG_DIR, "error_GLM_multiViews.txt"), "w", encoding="utf-8")
    client = ZhipuAiClient(api_key=API_KEY)

    # Collect all location (sub-directory) names.
    imgs_name_list = [
        item for item in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, item)) and not item.startswith(".")
    ]

    # Keep only locations that have not been processed yet.
    need_process_imgs = [img for img in imgs_name_list if need_process(OUTPUT_DIR, img)]
    print(f"To process: {len(need_process_imgs)}")
    print(f"Skipped (valid JSON exists): {len(imgs_name_list) - len(need_process_imgs)}")

    count = 0
    err_list = []
    for img_name in need_process_imgs:
        view_files = [img_name + suffix for suffix in VIEW_FILES]
        print(f"----------{count}----------")
        images = [Image.open(os.path.join(IMAGES_DIR, img_name, vf)) for vf in view_files]

        sg = None
        try:
            raw = invoke_with_image(client, QUERY, images)
            sg = json.loads(extract_json_format(raw))
        except Exception as e:
            err_list.append(img_name)
            log_error(f"Error processing {img_name}.json: {e}", error_file)

        # Save one merged scene graph per location (all four views combined).
        with open(os.path.join(OUTPUT_DIR, f"{img_name}.json"), "w", encoding="utf-8") as f:
            json.dump(sg, f, ensure_ascii=False, indent=4)

        print(f"Done: {img_name}")
        count += 1

    final_msg = f"\nFailed list: {err_list}"
    print(final_msg)
    print(final_msg, file=error_file)
    error_file.flush()
    error_file.close()


if __name__ == "__main__":
    main()
