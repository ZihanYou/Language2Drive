# Third-party notices

This archive distributes project tooling and documentation, not third-party
model implementations, checkpoint code, weights or dataset contents.

- Waymo Open Dataset / WOD-E2E: data protocol and reference index conventions.
  https://waymo.com/open/data/e2e/
  https://github.com/waymo-research/waymo-open-dataset
- DriveMA: external model and original WOD prompt interface adapted by
  `scripts/model_adapters/drivema.py`.
  https://github.com/Tsinghua-MARS-Lab/DriveMA
  https://huggingface.co/zwc2003/DriveMA-2B
- Fast-dDrive: external model, WOD prompt and Scaffold Spec generation interface
  adapted by `scripts/model_adapters/fast_ddrive.py`.
  https://github.com/NVlabs/Fast-dLLM/tree/main/fast_ddrive
  https://huggingface.co/Efficient-Large-Model/Fast-dDrive
- Qwen-Drive: external official runner and history interpolation loaded by
  `scripts/model_adapters/qwen_drive.py`; its source is obtained separately.
  https://github.com/QwenLM/Qwen-Drive-1.0
  https://huggingface.co/Qwen/Qwen-Drive-1.0-4B
- Pillow, PyTorch, Transformers, Hugging Face Hub and optional vLLM are external
  dependencies and retain their own licenses.

Upstream names and URLs identify public dependencies, not the anonymous project
authors. Project licensing does not grant access or redistribution rights to
external datasets or checkpoints, or change any upstream license obligations.
