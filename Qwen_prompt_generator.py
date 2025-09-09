'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2025-08-26 03:33:21
LastEditors: Mingxin Zhang
LastEditTime: 2025-09-10 01:43:19
Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
'''
import os
import torch
import gc
import json
import random
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
from pathlib import Path
from download import download_model

gc.collect(); torch.cuda.empty_cache()

# Script path
DIR = str(Path(__file__).resolve().parent)

def get_texture_folders(root_dir):
    return [os.path.join(root_dir, texture) for texture in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, texture))]

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
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_path + model_name, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_path + model_name, trust_remote_code=True)
    
    texture_folders = get_texture_folders(DIR + '/Texture/')
    for image_path in texture_folders:
        texture_name = image_path.split("/")[-1]
        # write json for prompts of each texture
        if os.path.exists(image_path + '/prompts.json'):
            with open(image_path + '/prompts.json', 'r') as f:
                prompts = json.load(f)
                
        else:
            prompts = {}
            prompts[str(texture_name)] = {}
            
        prompts[str(texture_name)][str(model_name)] = {}

        sampled_i = random.sample(range(1, 21), 5)
        for i in range(5): # generate 5 prompts for each texture
            query = tokenizer.from_list_format([
                # {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
                # {'text': '这是什么?'},
                {'image': image_path + '/' + str(sampled_i[i]) + '.jpg'},
                {'text': 'Carefully observe the surface texture shown in the image and describe its tactile characteristics in detail. \
                        Your description should include:\
                        •	Roughness level (e.g., smooth, fine, grainy, prickly)\
                        •	Macroscopic structure (e.g., patterned ridges, grooves, undulations)\
                        •	Microscopic structure (e.g., tiny granules, fuzziness, porous feel)\
                        •	Inferred tactile impression when pressing vertically on the shown surface or stroking the surface horizontally according to the material (such as soft, rigid)\
                        Generate an English passage that conveys the tactile features of this material, suitable for use in a haptic feedback generation task.'},
            ])
            response, history = model.chat(tokenizer, query=query, history=None)
            prompts[str(texture_name)][str(model_name)][str(i+1)] = response
        # save json
        with open(image_path + '/prompts.json', 'w') as f:
            json.dump(prompts, f, indent=4)
        print(f"Prompts for {texture_name} saved.")
        break