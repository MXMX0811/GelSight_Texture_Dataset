'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2025-08-26 03:29:22
LastEditors: Mingxin Zhang
LastEditTime: 2025-08-26 03:36:28
Copyright (c) 2025 by Mingxin Zhang, All Rights Reserved. 
'''

from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(repo_id: str, dest_dir: str, revision: str = "main", token: str | None = None) -> str:
    dest = Path(dest_dir)
    has_model_files = any((dest / f).exists() for f in ["config.json", "tokenizer.json", "tokenizer.model"]) \
                      or any(dest.glob("*.bin")) or any(dest.glob("*.safetensors"))

    if dest.is_dir() and has_model_files:
        print('Model exists.')
        return

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest,
        local_dir_use_symlinks=False,
        token=token
    ) 
