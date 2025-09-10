'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2025-08-26 03:33:21
LastEditors: Mingxin Zhang
LastEditTime: 2025-09-11 01:38:27
Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
'''
import os
import torch
import json
import random
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from pathlib import Path
from download import download_model

# Run pip install -e . in the dir models/Janus to install the Janus package first
# Script path
DIR = str(Path(__file__).resolve().parent)

def get_texture_folders(root_dir):
    return [os.path.join(root_dir, texture) for texture in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, texture))]

if __name__ == "__main__":
    # Download the model
    model_path = DIR + '/models/'
    model_name = 'Janus-Pro-7B'
    download_model(
        repo_id='deepseek-ai/Janus-Pro-7B',
        dest_dir=model_path + model_name,
        revision='main',
        token=os.getenv('HUGGING_FACE_HUB_TOKEN')
    )

    # Load
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path + model_name)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path + model_name, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    print(f"Model Parameters: {sum(p.numel() for p in vl_gpt.parameters()):,}")

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
            image = image_path + '/' + str(sampled_i[i]) + '.jpg'
            question = 'Carefully observe the surface texture shown in the image and describe its tactile characteristics.\
                        You should include detailed quantitative descriptions of the following aspects:\
                        •	Roughness level (e.g., smooth, fine, grainy, prickly)\
                        •	Macroscopic structure (e.g., patterned ridges, grooves, undulations)\
                        •	Microscopic structure (e.g., tiny granules, fuzziness, porous feel)\
                        •	Inferred tactile impression according to the material (such as soft, rigid)\
                        The description should be suitable for use in a haptic feedback generation task.\
                        Use accurate, objective and concise descriptions. Use short sentences.\
                        Avoid overly subjective language and unnecessary embellishments.\
                        Do not use subjects and introductory phrases at the beginning of the response.\
                        No filler.\
                        Use English to answer.'
                        
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)

            # # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # # run the model to get the response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
            )
            
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            prompts[str(texture_name)][str(model_name)][str(i+1)] = answer
            
        # save json
        with open(image_path + '/prompts.json', 'w') as f:
            json.dump(prompts, f, indent=4)
        print(f"Prompts for {texture_name} saved.")