import re

def extract_json_format(data_str):
    # 提取JSON部分（移除首尾标记）
    json_start = data_str.find('{')  # 找到第一个{
    json_end = data_str.rfind('}') + 1  # 找到最后一个}
    json_str = data_str[json_start:json_end]
    return json_str