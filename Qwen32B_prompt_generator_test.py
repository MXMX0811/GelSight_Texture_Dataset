'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2025-08-26 03:33:21
LastEditors: Mingxin Zhang
LastEditTime: 2025-09-03 16:57:49
Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
'''
import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from pathlib import Path
from download import download_model

# Script path
DIR = str(Path(__file__).resolve().parent)

if __name__ == "__main__":
    # Download the model
    model_path = DIR + '/models/'
    model_name = 'Qwen2.5-VL-32B-Instruct'
    download_model(
        repo_id='Qwen/Qwen2.5-VL-32B-Instruct',
        dest_dir=model_path + model_name,
        revision='main',
        token=os.getenv('HUGGING_FACE_HUB_TOKEN')
    )

    # Load
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path + model_name, torch_dtype="auto", device_map="cuda"
)
    processor = AutoProcessor.from_pretrained(model_path + model_name)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    'image': DIR + '/Texture/Leather/1.jpg'
                },
                {
                    "type": "text", 
                    "text": 'Carefully observe the surface texture of the material shown in the image and describe its tactile characteristics in detail. \
                             Your description should include:\
                             •	Roughness level (e.g., smooth, fine, grainy, prickly)\
                             •	Macroscopic structure (e.g., patterned ridges, grooves, undulations)\
                             •	Microscopic structure (e.g., tiny granules, fuzziness, porous feel)\
                             •	Inferred tactile impression (overall hardness or softness of the surface, such as soft, rigid, slightly elastic)\
                             Generate a passage that conveys the tactile experience of this material, suitable for use in a haptic feedback generation task.'
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)