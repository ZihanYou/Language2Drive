#!/usr/bin/env python3
"""DriveMA-2B two-turn inference on public WOD-E2E prerelease requests.

This is an experimental passenger-language adaptation of the official three-
camera prompt. The released model was trained with route-intent labels, not
arbitrary passenger commands. Predictions must be labelled accordingly.

Only public/inputs.jsonl, public/workload.jsonl and public JPEGs are opened.
The private reference sidecar is neither read nor accepted as an argument.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


VIEWS = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT")
POINT = re.compile(r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_requests(path: Path, args: argparse.Namespace) -> list[dict]:
    requests = read_jsonl(path)
    if args.limit is not None:
        requests = requests[:args.limit]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    return requests[args.shard_index::args.num_shards]


def safe_image(root: Path, item: object) -> Path:
    relative = item.get("path") if isinstance(item, dict) else item
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("image must have a relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or unsafe image: {relative}")
    return path


def history(row: dict) -> dict[str, list]:
    past = row["past_states"]
    for key in ("x", "y", "vx", "vy", "ax", "ay"):
        if key not in past or len(past[key]) != 16:
            raise ValueError(f"{row['input_id']}: past_states.{key} must have 16 points")
        if not all(math.isfinite(float(v)) for v in past[key]):
            raise ValueError(f"{row['input_id']}: non-finite past state")
    return {
        "traj": list(zip(past["x"], past["y"])),
        "vel": list(zip(past["vx"], past["vy"])),
        "acc": list(zip(past["ax"], past["ay"])),
    }


def pair(x: object) -> str:
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        raise ValueError("expected x/y pair")
    return f"[{float(x[0]):.2f}, {float(x[1]):.2f}]"


def pairs(seq: list) -> str:
    return ", ".join(pair(x) for x in seq)


def first_turn_text(row: dict, instruction: str) -> str:
    h = history(row)
    intent = row.get("route_command", "UNKNOWN")
    if intent not in ("GO_STRAIGHT", "GO_LEFT", "GO_RIGHT", "UNKNOWN"):
        raise ValueError(f"{row['input_id']}: unsupported route command {intent}")
    return (
        "You are an expert driver.\n"
        "Input:\n"
        "- 1 frame of multi-view images collected from the ego-vehicle at the present timestep: "
        "front_left_view: <image>; front_view:<image>; front_right_view:<image>\n"
        f"- Current high-level intent:{intent}\n"
        f"- Current acceleration is {pair(h['acc'][-1])}\n"
        f"- Current velocity is {pair(h['vel'][-1])}\n"
        f"- 4-second past trajectory (16 steps at 4 Hz):{pairs(h['traj'])}\n"
        f"- 4-second past acceleration (16 steps at 4 Hz):{pairs(h['acc'])}\n"
        f"- 4-second past velocity (16 steps at 4 Hz):{pairs(h['vel'])}\n"
        "Coordinate System Definition: X-axis: positive forward, negative backward; "
        "Y-axis: positive left, negative right.\n"
        f"Passenger instruction: {instruction}\n"
        "Task: Inspect the input and make the decision while following the passenger instruction when safe.\n"
        "Output format:\n"
        "longitudinal action: xx, lateral action: xx"
    )


def second_turn_text() -> str:
    return (
        "Task: Given the above information and the passenger instruction, predict the "
        "5-second future trajectory (5 steps at 1 Hz) of the ego vehicle.\n"
        "Output format:\n"
        "[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5]"
    )


def parse_five_points(text: str) -> list[list[float]]:
    matches = POINT.findall(text)
    if len(matches) != 5:
        raise ValueError(f"expected exactly 5 [x, y] points, got {len(matches)}")
    result = [[float(x), float(y)] for x, y in matches]
    if not all(math.isfinite(value) for xy in result for value in xy):
        raise ValueError("non-finite output point")
    return result


def interpolate_to_wod(points_1hz: list[list[float]]) -> list[list[float]]:
    if len(points_1hz) != 5:
        raise ValueError("DriveMA must return five 1 Hz points")
    nodes = [[0.0, 0.0]] + points_1hz
    out = []
    for quarter in range(1, 21):
        left = (quarter - 1) // 4
        fraction = (quarter - 4 * left) / 4
        out.append([
            nodes[left][axis] + fraction * (nodes[left + 1][axis] - nodes[left][axis])
            for axis in (0, 1)
        ])
    return out


def user_content(prompt: str) -> list[dict]:
    parts = prompt.split("<image>")
    content = []
    for index, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if index < len(parts) - 1:
            content.append({"type": "image"})
    return content


def load_views(row: dict, root: Path):
    from PIL import Image

    images = []
    for view in VIEWS:
        path = safe_image(root, row["images"][view])
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    return images


def front_camera_history_complete(row: dict) -> bool:
    """Public camera-temporal flag, distinct from 16-point ego state history."""
    nominal = (-1.5, -1.0, -0.5, 0.0)
    for view in VIEWS:
        frames = row.get("camera_history", {}).get(view, [])
        if len(frames) != 4:
            return False
        try:
            for frame, expected in zip(frames, nominal):
                if abs(float(frame["nominal_offset_s"]) - expected) > 1e-6:
                    return False
                if abs(float(frame["offset_s"]) - expected) > 0.100001:
                    return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def render(tokenizer, conversation: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def run_vllm(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    inputs = {row["input_id"]: row for row in read_jsonl(args.inputs)}
    requests = selected_requests(args.workload, args)
    model = args.model.resolve()
    if not (model / "config.json").is_file() or not (model / "model.safetensors").is_file():
        raise FileNotFoundError(f"incomplete DriveMA checkpoint: {model}")
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True,
                                               local_files_only=True)
    llm = LLM(model=str(model), dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=True,
              limit_mm_per_prompt={"image": 3},
              mm_processor_kwargs={"max_pixels": args.max_pixels,
                                   "min_pixels": args.min_pixels},
              trust_remote_code=True)
    decision_sampling = SamplingParams(temperature=0, max_tokens=args.max_decision_tokens)
    trajectory_sampling = SamplingParams(temperature=0, max_tokens=args.max_trajectory_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for offset in range(0, len(requests), args.batch_size):
            batch = requests[offset:offset + args.batch_size]
            first_inputs, contexts = [], []
            for request in batch:
                row = inputs[request["input_id"]]
                instruction = str(request["instruction"])
                if request.get("instruction_language") != "en" or not instruction.strip():
                    raise ValueError(f"{request['request_id']}: expected English instruction")
                content = user_content(first_turn_text(row, instruction))
                images = load_views(row, args.image_root)
                first_inputs.append({
                    "prompt": render(tokenizer, [{"role": "user", "content": content}]),
                    "multi_modal_data": {"image": images},
                })
                contexts.append((request, content, images))
            decisions = llm.generate(first_inputs, decision_sampling, use_tqdm=False)
            second_inputs = []
            for (request, content, images), decision in zip(contexts, decisions):
                decision_text = decision.outputs[0].text.strip()
                conversation = [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": decision_text},
                    {"role": "user", "content": [{"type": "text", "text": second_turn_text()}]},
                ]
                second_inputs.append({"prompt": render(tokenizer, conversation),
                                      "multi_modal_data": {"image": images}})
            trajectories = llm.generate(second_inputs, trajectory_sampling, use_tqdm=False)
            for (request, _, _), decision, trajectory in zip(contexts, decisions, trajectories):
                text = trajectory.outputs[0].text.strip()
                result = {"request_id": request["request_id"],
                          "coordinate_frame": "ego_at_anchor", "reference_point": "rear_axle",
                          "adapter": "drivema_2b_passenger_prompt_experimental_v1",
                          "used_views": list(VIEWS),
                          "front_camera_history_complete": front_camera_history_complete(
                              inputs[request["input_id"]]),
                          "decision_text": decision.outputs[0].text.strip(),
                          "trajectory_text": text}
                try:
                    native = parse_five_points(text)
                    result["native_timestamps_s"] = [1, 2, 3, 4, 5]
                    result["native_xy_m"] = native
                    result["timestamps_s"] = [quarter / 4 for quarter in range(1, 21)]
                    result["xy_m"] = interpolate_to_wod(native)
                    result["status"] = "ok"
                except ValueError as error:
                    result["status"] = "invalid_output"
                    result["error"] = str(error)
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()


def run_transformers(args: argparse.Namespace) -> None:
    """Use the released Qwen3.5 architecture through Hugging Face Transformers."""
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    inputs = {row["input_id"]: row for row in read_jsonl(args.inputs)}
    requests = selected_requests(args.workload, args)
    model_dir = args.model.resolve()
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"incomplete DriveMA checkpoint: {model_dir}")
    device = f"cuda:{args.device_index}"
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True,
        max_pixels=args.max_pixels, min_pixels=args.min_pixels,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()

    def generate(conversation: list[dict], max_new_tokens: int) -> str:
        prepared = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            tokens = model.generate(
                **prepared, do_sample=False, max_new_tokens=max_new_tokens,
            )
        return processor.batch_decode(
            [tokens[0, prepared.input_ids.shape[1]:]],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.is_file() else "w"
    completed = set()
    if mode == "a":
        completed = {row["request_id"] for row in read_jsonl(args.output)}
    with args.output.open(mode, encoding="utf-8") as output:
        for request in requests:
            if request["request_id"] in completed:
                continue
            row = inputs[request["input_id"]]
            instruction = str(request["instruction"])
            if request.get("instruction_language") != "en" or not instruction.strip():
                raise ValueError(f"{request['request_id']}: expected English instruction")
            images = load_views(row, args.image_root)
            pieces = user_content(first_turn_text(row, instruction))
            image_iterator = iter(images)
            content = [
                {**piece, "image": next(image_iterator)} if piece["type"] == "image" else piece
                for piece in pieces
            ]
            conversation = [{"role": "user", "content": content}]
            decision = generate(conversation, args.max_decision_tokens)
            conversation.extend([
                {"role": "assistant", "content": decision},
                {"role": "user", "content": [{"type": "text", "text": second_turn_text()}]},
            ])
            trajectory = generate(conversation, args.max_trajectory_tokens)
            result = {
                "request_id": request["request_id"],
                "coordinate_frame": "ego_at_anchor",
                "reference_point": "rear_axle",
                "adapter": "drivema_2b_passenger_prompt_experimental_v1",
                "backend": "transformers",
                "used_views": list(VIEWS),
                "front_camera_history_complete": front_camera_history_complete(row),
                "decision_text": decision,
                "trajectory_text": trajectory,
            }
            try:
                native = parse_five_points(trajectory)
                result.update({
                    "status": "ok",
                    "native_timestamps_s": [1, 2, 3, 4, 5],
                    "native_xy_m": native,
                    "timestamps_s": [quarter / 4 for quarter in range(1, 21)],
                    "xy_m": interpolate_to_wod(native),
                })
            except ValueError as error:
                result.update({"status": "invalid_output", "error": str(error)})
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True,
                        help="directory containing images/ in the public release")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-decision-tokens", type=int, default=128)
    parser.add_argument("--max-trajectory-tokens", type=int, default=160)
    parser.add_argument("--backend", choices=("transformers", "vllm"),
                        default="transformers")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    (run_transformers if args.backend == "transformers" else run_vllm)(args)


if __name__ == "__main__":
    main()
