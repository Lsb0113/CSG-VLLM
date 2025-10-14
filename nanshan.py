import base64
import os
import json
from PIL import Image
from zhipuai import ZhipuAI
import io
import pandas as pd
# import yaml


# import cv2

# from openpyxl import load_workbook
from funcset.extract_json import extract_nested_braces
from dataclasses import dataclass

@dataclass
class config:
    dataset:str='nanshan' # Flickr30k, MSCOCO
    use_seg_images:bool=False
    use_scene_graph:bool=False
    temperature:float=0.5
    top_p:float=0.7
    max_text_tokens:int = 1024
    max_scene_graph_tokens: int = 512


client = ZhipuAI(api_key="3251aca0da461fb23b1cadcb12088924.6ACY6stGAeYL43kq")  # 请填写您自己的API Key


def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def invoke_with_image(query, images_dict=None):
    image = Image.open(images_dict)
    base64_image = encode_image_to_base64(image)

    image_message = {"type": "image_url",
                     "image_url": {"url": base64_image}}

    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
    messages[0]['content'].append(image_message)

    response = client.chat.completions.create(model='GLM-4V-Plus-0111', messages=messages, max_tokens=1024,
                                              temperature=0.5, top_p=0.7)
    out = response.choices[0].message.content
    return out


def invoke_with_sg_text(query):
    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
    response = client.chat.completions.create(model='GLM-4-Air', messages=messages, max_tokens=512,
                                              temperature=0.5, top_p=0.7)
    out = response.choices[0].message.content
    return out


if __name__ == "__main__":
    config = config()
    if config.use_seg_images:
        imgs_path = 'nanshan/nanshan_m2f_cityscape/nanshan_seg_jpeg'

    # imgs_path ='nanshan/nanshan_m2f_cityscape/nanshan_seg_jpeg'
    imgs_path = 'E:\\3DSSG-cv21\VLLM2SSG\\new_data\\nanshan_jpeg'
    images_names = 'nanshan/seg_name_list_new.json'
    with open(images_names, 'r') as f:
        data_dict = json.load(f)

    names_list = data_dict.keys()

    # save_text_path = 'nanshan_2.xlsx'
    data = {}
    data_img_names = []
    data_img2txt = []

    # yaml_save_path = 'Flickr30k/flickr30k-images/ori_image_sg_np_1000/'

    file_path = 'nanshan_elder_des.csv'
    batch_size = 20
    num_images = len(names_list)

    max_batch_count = int(num_images / batch_size)

    # # original No SG template nanshan
    # template_str = """
    #     # 角色
    #     -你是一位优秀的图像分析专家。
    #
    #     ## 目标
    #     - 准确描述图像呈现的信息。
    #
    #     ## 规则
    #     -按照以下步骤执行：
    #     1. 识别图像中的各种对象，识别对象的显式特征（如：外形、颜色、材质）。
    #     2. 识别对象上的文字信息，然后描述对象的功能性（例如，医院的功能是医疗，餐厅的功能性是用餐，公园的功能是给人游玩。）。所以这一步需要你抽取对象的功能属性，这些功能属性是隐式特征。
    #     3. 识别这些对象在图像中的关系。
    #     4. 将对象信息、对象之间的关系和场景的功能性总结成一段【文本描述】，字数控制在200字以内。
    #
    #     ## 注意事项
    #     - 不需要识别行人、车辆(汽车，货车，公交车，自行车，电摩托等)、天空、树木等对象，如果【文本描述】中有这些对象，则去掉它们。
    #     - 充分提取图像中的信息。
    #     - 保证【文本描述】的逻辑连贯、规范，与图像内容保持一致。
    #
    #     ## 输出内容
    #     输出总结的【文本描述】即可，不要输出思考过程内容。
    #     """

    # template_sg = """
    #     # Role:你是一位优秀的图像分析专家。
    #
    #     ## Goals
    #     - 准确描述图像呈现的信息。
    #
    #     ## Constrains
    #     -按照以下步骤执行：
    #     1.识别图像中的对象，识别对象的外观特征，比如：颜色、材质、形状。
    #     2.识别对象的属性特征，比如：对象上的文字信息，对象的功能性（例如，医院的功能是医疗，餐厅的功能是餐饮，公园的功能是游玩。）。这一步可以根据你自身的知识推理对象可能的功能。
    #     3.识别这些对象在图像中的相对位置关系。
    #     4.然后以json格式输出节点和关系，输出格式如下：
    #     {nodes: [{id: 对象序号, name: 对象名称，feature：显式特征+隐式特征(使用一句话描述对象特征，可以根据你自身的知识合理地描述这两个特征)},...],
    #     relations: [{from: 起始节点id, to: 目标节点id, relation: 关系（不包含对象，例如：起始对象的左边是目标对象，关系保存为：左边）},...]}
    #
    #     ## Skills
    #     - 充分提取图像中的信息。
    #     - 如果图像中同一类物体太多，可以把它们归为一类。
    #     - 需要保证生成的json内容与图像内容保持一致。
    #
    #     ## Workflow
    #     不要解释，直接输出json格式内容。
    #     """

    # template_seg_obj = """
    #     # 角色:你是一位优秀的图像分析专家。
    #
    #     ## Goals
    #     - 准确描述图像呈现的信息。
    #
    #     ## Constrains
    #     -按照以下步骤执行：
    #     1.识别图像中的对象，识别对象的外观特征，比如：颜色、材质、形状。
    #     2.识别对象的属性特征，比如：对象上的文字信息，对象的功能性（例如，医院的功能是医疗，餐厅的功能是餐饮，公园的功能是游玩。）。这一步可以根据你自身的知识推理对象可能的功能。
    #     3.识别这些对象在图像中的相对位置关系。
    #     4.然后以json格式输出节点和关系，输出格式如下：
    #     {nodes: [{id: 对象序号, name: 对象名称，feature：显式特征+隐式特征(使用一句话描述对象特征，可以根据你自身的知识合理地描述这两个特征)},...],
    #     relations: [{from: 起始节点id, to: 目标节点id, relation: 关系（不包含对象，例如：起始对象的左边是目标对象，关系保存为：左边）},...]}
    #
    #     ## Skills
    #     - 充分提取图像中的信息。
    #     - 如果图像中同一类物体太多，可以把它们归为一类。
    #     - 需要保证生成的json内容与图像内容保持一致。
    #
    #     ## Workflow
    #     不要解释，直接输出json格式内容。
    #     """
    # prompt_temp = ChatPromptTemplate.from_template(template_str)

    #小孩描述
    template_str = """
        请你从老年人视角描述所给的图片的街景内容。描述字数控制在100字以内。
        """
    count = 0
    batch_count = 1

    # id_map = {}
    # id = 10001
    # precess_img_num = 5
    err_list = []
    new_data = []

    imgs_js_path = 'nanshan/nanshan_m2f_cityscape/js/nanshan_js/'

    for img_name in names_list:
        # if count not in err_list: # 未识别，识别出错
        #     count += 1
        #     continue

        # if count<380: #意外中断
        #     count+=1
        #     if (count+1) % batch_size == 0:
        #         batch_count += 1
        #     continue

        img_view_1 = img_name + "_0_0.jpeg"
        img_view_2 = img_name + "_90_0.jpeg"
        img_view_3 = img_name + "_180_0.jpeg"
        img_view_4 = img_name + "_270_0.jpeg"

        print(f'----------{count}-----------')
        # if count<2:
        #     count+=1
        #     continue
        # if count >= 5:  # 测试阶段控制输入
        #    break

        # 同一地点的不同视角图片
        # img_name = img_name[:-4] + ".jpeg"

        # 图像-》SG
        out1 = None
        out2 = None
        out3 = None
        out4 = None

        sg_1 = None
        sg_2 = None
        sg_3 = None
        sg_4 = None
        # print(Image.open(os.path.join(original_imgs_path, img_view_1)))
        try:
            out1 = invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_1))
            out2 = invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_2))
            out3 = invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_3))
            out4 = invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_4))
            # sg_1 = json.loads(extract_nested_braces(out1))
            # sg_2 = json.loads(extract_nested_braces(out2))
            # sg_3 = json.loads(extract_nested_braces(out3))
            # sg_4 = json.loads(extract_nested_braces(out4))
        except Exception as e:
            print(f"发生错误: {e}")
            err_list.append(count)
            print(out1)
            print(out2)
            print(out3)
            print(out4)

        # data = {}
        # data[img_view_1] = {'sg': sg_1, 'left': img_view_4, 'right': img_view_2}
        # data[img_view_2] = {'sg': sg_2, 'left': img_view_1, 'right': img_view_3}
        # data[img_view_3] = {'sg': sg_3, 'left': img_view_2, 'right': img_view_4}
        # data[img_view_4] = {'sg': sg_4, 'left': img_view_3, 'right': img_view_1}
        #
        # with open(f'{imgs_js_path}' + img_name + '.json', 'w',
        #           encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)

        new_data.append([img_name, out1 + out2 + out3 + out4])

        if len(new_data) == batch_size and batch_count <= max_batch_count:

            try:  # 加载现有的文件
                exist_table = pd.read_csv(file_path)
            except FileNotFoundError:  # 如果文件不存在，创建一个空表格并设置表头
                exist_table = pd.DataFrame(columns=['编号', '描述'])

            print(len(new_data))
            print(new_data)
            new_data_frame = pd.DataFrame(new_data, columns=['编号', '描述'])
            combine_data = pd.concat([exist_table, new_data_frame], ignore_index=True)
            combine_data.to_csv(file_path, index=False)
            print("数据已成功添加到 Excel 文件中。")
            batch_count += 1
            new_data = []

        if batch_count > max_batch_count and len(new_data) != 0:  # 将剩下的数据添加进去
            try:  # 加载现有的文件
                exist_table = pd.read_csv(file_path)
            except FileNotFoundError:  # 如果文件不存在，创建一个空表格并设置表头
                exist_table = pd.DataFrame(columns=['编号', '描述'])

            new_data_frame = pd.DataFrame(new_data, columns=['编号', '描述'])
            combine_data = pd.concat([exist_table, new_data_frame], ignore_index=True)
            combine_data.to_csv(file_path, index=False)
            print("数据已成功添加到 Excel 文件中。")
            new_data = []

        count += 1
    print(err_list)
