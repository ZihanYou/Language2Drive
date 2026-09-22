import io
import json
import base64
from typing import List, Dict, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Base64Bytes
from PIL import Image
import requests
import uvicorn
import argparse

from sglang.utils import launch_server_cmd, wait_for_server
import os


def start_qwen_server(qwen_ckpt_path):

    vision_process, port = launch_server_cmd(
        f"""
        /path/to/your/envs/sglang/bin/python -m sglang.launch_server \
            --model-path {qwen_ckpt_path} \
            --chat-template=qwen2-vl
        """
    )

    # save port
    port_file_dir = os.path.dirname(args.port_file)
    if port_file_dir and not os.path.exists(port_file_dir):
        os.makedirs(port_file_dir, exist_ok=True)
        print(f"✅ Created directory: {port_file_dir}")

    with open(f"{args.port_file}", "w") as f:
        f.write(str(port))

    wait_for_server(f"http://localhost:{port}")
    print(f"✅ Qwen service ready at port {port}")
    return vision_process, port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Qwen server with optional checkpoint path."
    )
    parser.add_argument(
        "--qwen_ckpt_path", type=str, required=True, help="Path to Qwen checkpoint."
    )
    parser.add_argument(
        "--port_file", type=str, required=True, help="Path to port file."
    )
    args = parser.parse_args()

    _, QWEN_PORT = start_qwen_server(args.qwen_ckpt_path)
