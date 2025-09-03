'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2025-08-26 03:33:21
LastEditors: Mingxin Zhang
LastEditTime: 2025-09-03 18:33:33
Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
'''
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
from pathlib import Path
from download import download_model

# Script path
DIR = str(Path(__file__).resolve().parent)

if __name__ == "__main__":
    # Download the model
    model_path = DIR + '/models/'
    model_name = 'Qwen-VL-Chat'
    download_model(
        repo_id='Qwen/Qwen-VL-Chat',
        dest_dir=model_path + model_name,
        revision='main',
        token=os.getenv('HUGGING_FACE_HUB_TOKEN')
    )

    # Load
    model = AutoModelForCausalLM.from_pretrained(model_path + model_name, trust_remote_code=True).eval()
    model = model.to(device='cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_path + model_name, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_path + model_name, trust_remote_code=True)
    
    query = tokenizer.from_list_format([
        # {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
        # {'text': '这是什么?'},
        {'image': DIR + '/Texture/Carpet/1.jpg'},
        {'text': 'Carefully observe the surface texture of the material shown in the image and describe its tactile characteristics in detail. \
                  Your description should include:\
                  •	Roughness level (e.g., smooth, fine, grainy, prickly)\
                  •	Macroscopic structure (e.g., patterned ridges, grooves, undulations)\
                  •	Microscopic structure (e.g., tiny granules, fuzziness, porous feel)\
                  •	Inferred tactile impression (overall hardness or softness of the surface, such as soft, rigid, slightly elastic)\
                  Generate a passage that conveys the tactile features of this material, suitable for use in a haptic feedback generation task.'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)