import re
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

import os

from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from datetime import datetime


NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


# === Simulate UniAD API Schema ===
class Calibration(BaseModel):
    """Calibration data."""

    camera2image: Dict[str, List[List[float]]]
    """Camera intrinsics. The keys are the camera names."""
    camera2ego: Dict[str, List[List[float]]]
    """Camera extrinsics. The keys are the camera names."""
    lidar2ego: List[List[float]]
    """Lidar extrinsics."""


class InferenceInputs(BaseModel):
    """Input data for inference."""

    images: Dict[str, Base64Bytes]
    """Camera images in PNG format. The keys are the camera names."""
    ego2world: List[List[float]]
    """Ego pose in the world frame."""
    canbus: List[float]
    """CAN bus signals."""
    timestamp: int  # in microseconds
    """Timestamp of the current frame in microseconds."""
    command: Literal[0, 1, 2]
    """Command of the current frame."""
    calibration: Calibration
    """Calibration data."""


class InferenceAuxOutputs(BaseModel):
    objects_in_bev: Optional[List[List[float]]] = None
    object_classes: Optional[List[str]] = None
    object_scores: Optional[List[float]] = None
    object_ids: Optional[List[int]] = None
    future_trajs: Optional[List[List[List[List[float]]]]] = None


class InferenceOutputs(BaseModel):
    trajectory: List[List[float]]
    aux_outputs: InferenceAuxOutputs


def decode_torch_image(byte_data: bytes) -> np.ndarray:
    # torch.save -> torch.load -> numpy
    tensor = torch.load(io.BytesIO(byte_data)).clone()
    return tensor.numpy()


def encode_image(img: np.ndarray) -> str:
    img_pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def global_to_ego(ego2world, p_global):
    """
    ego2world: 4x4 numpy array
    p_global: (3,) numpy array (global xyz)

    return: (3,) numpy array, ego frame xyz
    """
    R = ego2world[:3, :3]  # rotation matrix
    t = ego2world[:3, 3]  # translation vector
    p_ego = R.T @ (p_global - t)  # apply inverse transform
    return p_ego


def extract_trajectory(answer: str) -> List[List[float]]:
    # Try to extract [[x1,y1], [x2,y2], ...]
    try:
        matches = re.findall(r"\[([\-\d\.]+),\s*([\-\d\.]+)\]", answer)
        traj = [[float(x), float(y)] for x, y in matches]
        if len(traj) == 0:
            print("⚠️ Failed to extract trajectory from response, returning default 0")
            return [[0.0, 0.0]] * 6
        return traj[:6]  # Limit to 6 steps
    except Exception as e:
        print(f"Trajectory parsing error: {e}")
        return [[0.0, 0.0]] * 6


def format_ego_status(can) -> str:

    if len(can) < 18:
        raise ValueError(
            "Canbus signal too short; expected at least 17 values.")

    x, y = can[0], can[1]
    accel_x, accel_y = can[7], can[8]
    velocity_x, velocity_y = can[13], can[14]
    velocity = (velocity_x**2 + velocity_y**2) ** 0.5
    steering_angle = can[16]  # patch angle (弧度制)

    return (
        # f"[{x:.2f}, {y:.2f}], "
        f"Acceleration: X {accel_x:.2f}, Y {accel_y:.2f} m/s^2, "
        f"Velocity: {velocity:.2f} m/s "
        # f"Steering angle: {steering_angle:.2f} "
        # f"(positive: left turn, negative: right turn)"
    )


def preproc_canbus(ego_status):
    """Preprocesses the raw canbus signals from nuscenes."""
    rotation = Quaternion(ego_status[3:7])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180
    if patch_angle < 0:
        patch_angle += 360
    # extend the canbus signals with the patch angle, first in radians then in degrees
    new_ego_status = np.append(ego_status, patch_angle / 180 * np.pi)
    new_ego_status = np.append(new_ego_status, patch_angle)
    # UniAD has this, which is faulty as it does not copy the four elements, but just the first one, but we follow it for now
    new_ego_status[3:7] = -rotation
    return new_ego_status


def bytestr_to_numpy(images: Dict[str, Base64Bytes]) -> Dict[str, np.ndarray]:
    """Convert a list of png bytes to a numpy array of shape (n, h, w, c)."""
    result = {}
    for cam_name, img_bytes in images.items():
        try:
            # image_bytes = base64.b64decode(img_bytes)
            tensor = torch.load(io.BytesIO(img_bytes),
                                weights_only=False).clone()
            result[cam_name] = tensor.numpy()
        except Exception as e:
            print(f"❌ Failed to decode {cam_name}: {e}")
    return result


def numpy_image_to_base64(img_np: np.ndarray) -> str:
    """
    Convert numpy image directly to base64 string, without saving to disk.
    img_np: numpy array (H, W, C), dtype=uint8
    """
    img_np = img_np.astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


# === launch server ===
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9000)
# parser.add_argument("--config_path", type=str, required=True)
# parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--qwen_infer_port", type=int, required=True)
parser.add_argument("--past_pos_path", type=str, required=True)
parser.add_argument("--ego_status_path", type=str, required=True)
parser.add_argument("--qwen_ckpt_path", type=str, required=True)


args = parser.parse_args()


def ensure_dir_exists(file_path):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")


ensure_dir_exists(args.past_pos_path)
ensure_dir_exists(args.ego_status_path)

app = FastAPI()
# _, QWEN_PORT = start_qwen_server()
QWEN_PORT = args.qwen_infer_port

ego_status_path = args.ego_status_path
# file_path = args.past_pos_path
file_path = (
    args.past_pos_path
)  # This should be set to 'path to your past_pos.npy' or similar


@app.get("/alive")
async def alive() -> bool:
    return True


@app.post("/reset")
async def reset_runner() -> bool:
    open(ego_status_path, "w").close()
    np.save(file_path, np.empty((0, 3)))  # Save an empty array
    return True


@app.post("/infer")
async def infer(data: InferenceInputs) -> InferenceOutputs:

    imgs_dict = bytestr_to_numpy(data.images)

    cam_front = imgs_dict["CAM_FRONT"]
    cam_front_right = imgs_dict["CAM_FRONT_RIGHT"]
    cam_front_left = imgs_dict["CAM_FRONT_LEFT"]
    base64_image = numpy_image_to_base64(cam_front)
    right_base64_image = numpy_image_to_base64(
        cam_front_right
    )
    left_base64_image = numpy_image_to_base64(
        cam_front_left
    )

    canbus_signal = np.array(data.canbus)
    canbus_signal = preproc_canbus(canbus_signal)
    current_pos = canbus_signal[0:3]  # Take the first three values

    # If the file exists, read historical data
    if os.path.exists(file_path):
        past_positions = np.load(file_path)
    else:
        past_positions = np.empty((0, 3))  # Initialize an empty array

    # Add current data
    updated_positions = np.vstack([past_positions, current_pos])

    # Save back to file
    np.save(file_path, updated_positions)  # Save as .npy file

    print(
        f"Current canbus_signal[0:3] added, new shape is {updated_positions.shape}")

    canbus_position = global_to_ego(
        np.array(data.ego2world), canbus_signal[:3])
    canbus_signal[:3] = canbus_position

    with open(ego_status_path, "r") as f:
        lines = f.readlines()

    lines = [line.rstrip("\n") for line in lines]
    time_period = len(lines) * 0.5

    # Construct prompt
    fixed_prompt = f"You are an autonomous driving agent. You have access to front view camera image <image>, front left view camera image <image>, front right view camera image <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given the vehicle's intent inferred from the images.Provided are the previous ego vehicle status recorded over the last {time_period:.1f} seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format [x, y]:."

    new_canbus_info = format_ego_status(canbus_signal)
    lines.append(new_canbus_info)

    with open(ego_status_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    status_history = ""
    total_frames = min(len(lines), 6)
    for i, (pos, line) in enumerate(zip(updated_positions[-total_frames:], lines)):
        # The current frame is the earliest t-(n-1)*0.5s, the last is t-0.0s
        time_offset = (total_frames - 1 - i) * 0.5
        ego_pos = global_to_ego(np.array(data.ego2world), pos)
        x, y = ego_pos[0], ego_pos[1]
        status_history += f"(t-{time_offset:.1f}s) [{x:.1f}, {y:.1f}], {line}, "

    final_prompt = fixed_prompt + status_history.rstrip(", ")

    qwen_payload = {
        "model": args.qwen_ckpt_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{right_base64_image}"
                        },
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{left_base64_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 512,
    }

    # Call Qwen API
    qwen_response = requests.post(
        f"http://localhost:{QWEN_PORT}/v1/chat/completions", json=qwen_payload
    )
    qwen_response.raise_for_status()
    answer = qwen_response.json()["choices"][0]["message"]["content"]

    # print("[Qwen response]:", answer)

    # Extract coordinates from natural language (example)
    trajectory = extract_trajectory(answer)

    return InferenceOutputs(
        trajectory=trajectory, aux_outputs=InferenceAuxOutputs()  # Fill as needed
    )


uvicorn.run(app, port=args.port)
