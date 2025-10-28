import os
from PIL import Image
import json
from datasets import Dataset,Image

def data_process_template(instruction, image, answer):
    template = [
        {"role": "user",
         "content": [{"type": "text", "text": instruction}, {"type": "image", "image": image}]
         },
        {"role": "assistant",
         "content": [{"type": "text", "text": answer}]
         },
    ]
    return template


instruction = f"""
    ## Role
    You are a professional expert in structured image analysis, skilled at accurately extracting object features and relationships from images.
    ## Goals
    - Based on the image content, generate structured JSON data that includes "object features" and "inter-object relationships", ensuring the information is complete and logically rigorous.
    ##Constraints
    - Strictly follow the four steps below, with the output of each step providing a basis for the next:
        1. **Object and Appearance Feature Recognition**
            - List all core objects in the image (main objects within the object segmentation contour lines).
            - Describe the explicit appearance features of each object: color, material, and shape.
            - Objects of the same type (e.g., 3 white cars of the same model) can be grouped into a single object, with the quantity marked (e.g., "White Car × 3").
        2. **Object Attribute and Function Inference**
            -Extract the explicit attributes of objects: text on the surface (e.g., "Speed Limit 60" on a traffic sign), status (e.g., "Open Door", "Extinguished Light").
            -Infer implicit functions based on common sense (must conform to scene logic): for example, "Hospital" is inferred to "provide medical services", and "Fire Extinguisher" is inferred to "be used for fire fighting".
        3. **Inter-Object Relationship Recognition**
            - Only label direct spatial relationships (e.g., "left of", "above", "inside", "adjacent to") and functional associations (e.g., "control", "support", "contain").
            - Relationship descriptions should be concise (2-3 words), such as "above", "inside", "control". Do not include object names (Incorrect: "A is left of B" → Correct: "left of").
        4. **JSON Format Output**
            - Nodes must include: id (numeric sequence number, starting from 1), name (object name, with quantity marked if applicable, e.g., "White Car × 3"), feature (a single sentence combining explicit features and implicit functions, e.g., "Red plastic material, cylindrical shape, used for holding liquids").
            - Relationships must include: from (ID of the starting node), to (ID of the target node), relation (relationship term, strictly corresponding to Step 3).
            - Ensure all nodes appear at least once in the relationships (avoid isolated nodes).
    ## Skills
        - Prioritize retaining "high-information objects" in the image (e.g., traffic lights, store signs) and downplay low-information objects (e.g., blank walls).
    ## Workflow
        - Without any explanations or prefixes, directly output the JSON content that meets the format requirements (ensure there are no syntax errors and it can be parsed directly).
    """

datasets = {}
images_path = 'nanshan/nanshan_m2f_cityscape/nanshan_seg_jpeg'
images_list = os.listdir(images_path)

answer_json_path = 'nanshan/nanshan_seg/js_eng_single_files'


def convert_to_conversation(instruction, image_path, answer):
    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction},
             {"type": "image", "image": Image.open(image_path)}]
         },
        {"role": "assistant",
         "content": [
             {"type": "text", "text": answer}]
         },
    ]
    return {"messages": conversation}

# conversation_list = []
# for i in images_list:
#     name = i[:-4]
#     image_path = os.path.join(images_path, i)
#     # img = Image.open(image_path)
#     with open(os.path.join(answer_json_path, name + '.json'), 'r',encoding='utf-8') as f:
#         answer = json.load(f)
#     conversation_list.append(convert_to_conversation(instruction, image_path, str(answer)))

img_list=[]
text_list=[]

conversation_list = []
for img_name in images_list:
    image_path = os.path.join(images_path, img_name)
    img_list.append(image_path)
    with open(os.path.join(answer_json_path, img_name[:-5] + '.json'), 'r',encoding='utf-8') as f:
        answer = json.load(f)
    text_list.append(str(answer))

data = {
    "image": img_list,
    "text": text_list
}

# 构建数据集
dataset = Dataset.from_dict(data)

# 查看数据集信息（应显示 features: ['image', 'text'], num_rows: 68686）
print(dataset)

dataset = dataset.cast_column("image", Image())

# 按 8:2 划分训练集和验证集
dataset_split = dataset.train_test_split(test_size=0.2)
print(dataset_split)  # 输出包含 train 和 test 两个子集

# # 保存数据集（推荐用 parquet 格式，高效存储图像路径和文本）
# dataset.save_to_disk("./SFTData")  # 保存到文件夹
#
# # 后续加载
# from datasets import load_from_disk
# loaded_dataset = load_from_disk("./SFTData")