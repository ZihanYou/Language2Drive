"""Small diagnostic videos from all authentic camera JPEGs, without hashes."""
import math
from pathlib import Path
import subprocess


def write_review_video(frames, output, fps=10):
    """Require regular source timing; never silently hide gaps or invent frames."""
    output = Path(output)
    if output.exists():
        raise ValueError("preserve existing review video")
    if not math.isfinite(fps) or fps <= 0 or len(frames) < 2:
        raise ValueError("invalid video frame rate or missing source frames")
    timestamps = [int(timestamp) for timestamp, _ in frames]
    period_us = 1e6 / fps
    if any(abs((b - a) - period_us) > 2 for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("camera timestamps are irregular or contain missing frames")
    if any(not payload.startswith(b"\xff\xd8") for _, payload in frames):
        raise ValueError("expected original JPEG payloads")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(fps), "-i", "pipe:0",
        "-vf", "scale=576:320:flags=bicubic", "-an", "-c:v", "libx264", "-threads", "1",
        "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(output),
    ], input=b"".join(payload for _, payload in frames), capture_output=True, check=True, timeout=60)
    return {"file": output.name, "size_bytes": output.stat().st_size, "fps": fps,
            "frame_count": len(frames), "timestamps_us": timestamps,
            "purpose": "lossy_visual_review_only_not_model_input_or_support_attestation"}
