"""
Single-view scene graph generation using Zhipu GLM-4.6V-Flash API.

This script sends street-view images to the GLM-4.6V-Flash multimodal model
and extracts structured semantic scene graphs (nodes + relations) as JSON.

Usage:
    python CSG_SingleView.py

Before running, set your API key:
    export ZHIPU_API_KEY="your-api-key-here"
"""

import base64
import io
import json
import os
import sys

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from zai import ZhipuAiClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ZHIPU_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "GLM-4.6V-Flash"
IMAGES_DIR = "Datasets/perspective_1024/2021/nanshan"
OUTPUT_DIR = "Outputs/GLM/CSG_singleview"
LOG_DIR = "logs"
MAX_IMAGES = 0  # 0 = process all images

# ---------------------------------------------------------------------------
# Entity label vocabulary (Chinese labels as output by the GLM model)
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

# Prompt sent to the vision-language model (Chinese for best GLM performance)
SYSTEM_QUERY = f"""
你是计算机视觉领域的语义场景图专家，精通二维语义场景图的标准化构建。请基于我提供的街景/全景图片，生成一份结构完整、逻辑严谨、完全贴合画面的二维语义场景图，要求如下：

### 一、核心结构要求
1.  严格遵循「节点(nodes) + 关系(relations)」的JSON格式，不得出现格式错误、字段缺失
2.  节点(nodes)要求：
    - 每个节点必须包含：id：从1开始递增唯一整数。 type：实体类别，实体类别仅限以下范围：{ENTITY_ID_DICT}。type_id:实体类别ID,从实体字典里取。feature：必须包含外观特征：颜色、形状、材质、结构细节。 function：实体功能，简洁准确。text：提取实体表面可见文字（比如路名、招牌），无则为空字符串。
    - **强制要求**：绝对不提取行人、骑手、汽车、自行车这类动态、易变化的实体，仅提取固定不变的静态场景元素
    - 必须完整识别画面中所有核心静态实体，不得遗漏关键静态元素
3.  关系(relations)要求：
    - 每个关系必须包含：from：起点节点id、to：终点节点id、"Spatial_relation": "空间位置关系，仅限以下范围：{SPATIAL_MAP}", "Semantic_relation": "功能语义（包括但不限于如：指挥、引导、指示、服务、防护、装饰、照明、分隔、交通控制、停放、行驶、衬托、排列、控制等）", "Topological_relation": "拓扑关系,仅限以下范围：{TOPO_MAP}"
    - 空间关系、拓扑关系严格符合画面及上述限定范围，功能关系符合现实逻辑
    - 仅保留静态实体之间的关联关系，确保场景逻辑闭环

### 二、输出要求
1.  严格控制输出长度：仅提取核心、必要的静态实体，避免过度细分冗余实体，确保JSON完整输出，无截断
2.  优先保证格式完整：必须保证JSON结构完整、语法正确，可直接解析
3.  禁止任何额外内容：仅输出标准JSON，不得添加任何说明、注释、解释性文字，不得截断JSON结构
4.  提取实体数量控制在15个以内，可以少于15个实体，根据场景可提取实体数量而定。

### 三、质量要求
1.  100%贴合画面：所有静态实体、特征、位置关系完全对应图片内容，无捏造
2.  语义精准：实体分类、关系描述符合学术规范，严格遵循上述实体类别、空间关系、拓扑关系的限定，可直接用于模型输入
3.  结构完整：无遗漏核心静态实体、无缺失关键关系、无格式错误
"""


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def invoke_with_image(client: ZhipuAiClient, query: str, image_path: str) -> str:
    image = Image.open(image_path)
    base64_image = encode_image_to_base64(image)

    image_message = {
        "type": "image_url",
        "image_url": {"url": base64_image},
    }
    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
    messages[0]["content"].append(image_message)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=8192,
        temperature=0.5,
        top_p=0.7,
    )
    return response.choices[0].message.content


def log_error(msg: str, error_file) -> None:
    print(msg)
    print(msg, file=error_file)
    error_file.flush()


def need_process(output_dir: str, img_name: str) -> bool:
    """Check whether an image still needs scene-graph extraction."""
    base = os.path.splitext(img_name)[0]
    json_path = os.path.join(output_dir, base + ".json")
    if not os.path.exists(json_path):
        return True
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return not data
    except (json.JSONDecodeError, OSError):
        return True


def extract_json_format(data_str: str) -> str:
    """Extract the JSON object from model output (strip code fences)."""
    start = data_str.find("{")
    end = data_str.rfind("}") + 1
    return data_str[start:end]


def main() -> None:
    if API_KEY == "YOUR_API_KEY_HERE":
        sys.exit("Error: please set ZHIPU_API_KEY environment variable.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    error_file = open(os.path.join(LOG_DIR, "error_singleview.txt"), "w", encoding="utf-8")

    client = ZhipuAiClient(api_key=API_KEY)

    all_images = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    if MAX_IMAGES > 0:
        all_images = all_images[:MAX_IMAGES]

    pending = [img for img in all_images if need_process(OUTPUT_DIR, img)]
    print(f"Total images: {len(all_images)}")
    print(f"To process:    {len(pending)}")
    print(f"Skipped (already done): {len(all_images) - len(pending)}")

    count = 0
    errors = []
    for img_name in pending:
        print(f"[{count}] Processing {img_name}")
        sg = None
        try:
            raw = invoke_with_image(
                client,
                SYSTEM_QUERY,
                os.path.join(IMAGES_DIR, img_name),
            )
            sg = json.loads(extract_json_format(raw))
        except Exception as e:
            errors.append(img_name)
            log_error(f"Error processing {img_name}: {e}", error_file)

        out_name = os.path.splitext(img_name)[0] + ".json"
        with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as f:
            json.dump(sg, f, ensure_ascii=False, indent=4)
        count += 1

    final_msg = f"\nFailed list: {errors}"
    print(final_msg)
    print(final_msg, file=error_file)
    error_file.close()


if __name__ == "__main__":
    main()
