import base64
import os
import json
from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer
import io
import pandas as pd

from utils import extract_json_format
from dataclasses import dataclass

@dataclass
class config:
    use_seg_images:bool=False
    use_scene_graph:bool=False
    temperature:float=0.5
    top_p:float=0.7
    max_text_tokens:int = 1024
    max_scene_graph_tokens: int = 512


model, tokenizer = FastVisionModel.from_pretrained(
    "SHIBIN99/CSG-VLLM",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

class SavingTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.generated_text = ""
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        # super().on_finalized_text(text, stream_end)
        self.generated_text += text
        
def invoke_with_image(input_text, img_path=None):
    image=Image.open(img_path)
    inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
    ).to("cuda")
    
    text_streamer = SavingTextStreamer(tokenizer, skip_prompt=True)
    
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024,
                       use_cache = True, temperature = 0.5, min_p = 0.1)
    str_json=text_streamer.generated_text
    return str_json

if __name__ == "__main__":
    config = config()
    if config.use_seg_images:
        imgs_path = 'Data/Nanshan/seg_imgs'
    else:
        imgs_path = 'Data/Nanshan/ori_imgs'

    imgs_names = 'Datasets/Nanshan/nanshan_location_names.json'
    with open(imgs_names, 'r') as f:
        data_dict = json.load(f)

    names_list = data_dict.keys()

    file_path = 'nanshan_elder_des.csv'
    batch_size = 20
    num_images = len(names_list)

    max_batch_count = int(num_images / batch_size)

    # Prompt
    template_str = f"""
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
    
    count = 0
    img_js_path = 'Datasets/nanshan/js_files/'

    for img_name in names_list:

        img_view_1 = img_name + "_0_0.jpeg"
        img_view_2 = img_name + "_90_0.jpeg"
        img_view_3 = img_name + "_180_0.jpeg"
        img_view_4 = img_name + "_270_0.jpeg"

        print(f'----------{count}-----------')

        out1 = None
        out2 = None
        out3 = None
        out4 = None

        sg_1 = None
        sg_2 = None
        sg_3 = None
        sg_4 = None
        
        try:
            sg_1 = json.loads(extract_nested_braces(invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_1)))
            sg_2 = json.loads(extract_nested_braces(invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_2)))
            sg_3 = json.loads(extract_nested_braces(invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_3)))
            sg_4 = json.loads(extract_nested_braces(invoke_with_image(query=template_str, images_dict=os.path.join(imgs_path, img_view_4)))
        except Exception as e:
            print(f"An error occurred: {e}")
            err_list.append(count)
            print(sg_1)
            print(sg_2)
            print(sg_3)
            print(sg_4)

        data = {}
        data[img_view_1] = {'sg': sg_1, 'left': img_view_4, 'right': img_view_2}
        data[img_view_2] = {'sg': sg_2, 'left': img_view_1, 'right': img_view_3}
        data[img_view_3] = {'sg': sg_3, 'left': img_view_2, 'right': img_view_4}
        data[img_view_4] = {'sg': sg_4, 'left': img_view_3, 'right': img_view_1}
        
        with open(f'{img_js_path}' + img_name + '.json', 'w',
                  encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        count += 1
    print(err_list)

