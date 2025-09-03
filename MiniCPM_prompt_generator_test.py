'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2025-08-26 03:33:21
LastEditors: Mingxin Zhang
LastEditTime: 2025-09-03 19:14:15
Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
'''
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from download import download_model

# Script path
DIR = str(Path(__file__).resolve().parent)

if __name__ == "__main__":
    # Download the model
    model_path = DIR + '/models/'
    model_name = 'MiniCPMv2_6-prompt-generator'
    download_model(
        repo_id='pzc163/MiniCPMv2_6-prompt-generator',
        dest_dir=model_path + model_name,
        revision='main',
        token=os.getenv('HUGGING_FACE_HUB_TOKEN')
    )

    # Load
    model = AutoModel.from_pretrained(model_path + model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True).eval()
    model = model.to(device='cuda')
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_path + model_name, trust_remote_code=True)

    # Test
    image_path = DIR + '/Texture/Wallpaper2/'
    image = Image.open(image_path + '1.jpg').convert('RGB')
    question = 'Carefully observe the surface texture of the material shown in the image and describe its tactile characteristics in detail. \
                Your description should include:\
                •	Roughness level (e.g., smooth, fine, grainy, prickly)\
                •	Macroscopic structure (e.g., patterned ridges, grooves, undulations)\
                •	Microscopic structure (e.g., tiny granules, fuzziness, porous feel)\
                •	Inferred tactile impression (overall hardness or softness of the surface, such as soft, rigid, slightly elastic)\
                Generate a passage that conveys the tactile features of this material, suitable for use in a haptic feedback generation task.'
    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )
    print(res)