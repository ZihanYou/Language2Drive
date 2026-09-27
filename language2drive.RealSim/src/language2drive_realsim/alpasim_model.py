"""Language-conditioned AlpaSim model plugins backed only by private sidecar text."""

import threading
from typing import Any, Optional


try:
    from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model
except ImportError:  # Allows the benchmark core to run outside AlpaSim's Python 3.12 env.
    Alpamayo15Model = object  # type: ignore

try:
    from alpasim_driver.models.alpamayo2_model import (
        Alpamayo2Model,
        helper as _alpamayo2_helper,
    )
    from alpamayo2_super.chat_template.conversation import build_conversation
except ImportError:  # Same lightweight-import behavior as the A1.5 adapter.
    Alpamayo2Model = object  # type: ignore
    _alpamayo2_helper = None
    build_conversation = None


_ALPAMAYO2_HELPER_LOCK = threading.RLock()


def _alpamayo2_language_messages(
    data: dict[str, Any], model_config: Any, instruction: str
) -> list[dict[str, Any]]:
    """Build the official A2 navigation prompt with evaluator-sidecar text."""

    if build_conversation is None:
        raise RuntimeError("Alpamayo 2 Super dependencies are unavailable")
    text = str(instruction).strip()
    if not text:
        raise ValueError("Language2Drive sidecar instruction must be non-empty")
    conditioned = dict(data)
    conditioned["nav_text"] = [text]
    messages = build_conversation(
        data=conditioned,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        num_tokens_per_future_traj=model_config.tokens_per_future_traj,
        components_order=["image", "traj_history", "nav_instruction", "prompt"],
        components_prompt=["cot", "traj_future"],
        generation_mode=True,
        include_camera_ids=model_config.include_camera_ids,
        camera_ids=conditioned["camera_indices"],
        include_frame_nums=model_config.frame_label == "frame_num",
    )
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    return messages


class LanguageConditionedAlpamayo15Model(Alpamayo15Model):  # type: ignore
    """Pinned A1.5 adapter; the service attaches only free-form text to input.

    The upstream implementation derives ``nav_text`` from route geometry. This
    subclass deliberately ignores that value and inserts the trusted sidecar
    instruction at the same model chat boundary. The route remains available
    to the simulator as a common envelope but cannot answer the arm contrast.
    """

    _l2d_instruction: Optional[str] = None

    def predict(self, prediction_input: Any) -> Any:
        if not hasattr(prediction_input, "language_instruction"):
            raise ValueError("Language2Drive prediction input is missing sidecar text")
        self._l2d_instruction = str(prediction_input.language_instruction)
        try:
            return super().predict(prediction_input)
        finally:
            self._l2d_instruction = None

    def _create_chat_message(self, image_frames: Any, nav_text: str) -> Any:
        instruction = self._l2d_instruction
        if instruction is None:
            raise RuntimeError("language instruction is not active")
        return self._helper.create_message(
            image_frames.flatten(0, 1),
            self._camera_indices(),
            num_frames_per_camera=self._context_length,
            nav_text=instruction,
        )


class LanguageConditionedAlpamayo2Model(Alpamayo2Model):  # type: ignore
    """Pinned A2 adapter using its official raw navigation-instruction slot.

    Upstream ``Alpamayo2Model.predict`` calls the release helper's
    ``prepare_model_inputs`` directly. The helper in turn calls the module-level
    ``create_messages`` function. At this pinned commit there is no instance
    hook for that step, so this adapter replaces only that function while the
    single model call is active and restores it in ``finally``. AlpaSim's base
    A2 implementation evaluates a batch sequentially; the process-wide lock
    also prevents cross-session prompt mixing if another caller invokes the
    model concurrently.

    The prompt layout matches NVIDIA's released two-GPU navigation example:
    ``image, traj_history, nav_instruction, prompt``. Classifier-free guidance
    remains disabled because the standard AlpaSim driver exposes one-device
    placement; this is language conditioning, not the separate two-GPU CFG
    demonstration.
    """

    def predict(self, prediction_input: Any) -> Any:
        if not hasattr(prediction_input, "language_instruction"):
            raise ValueError("Language2Drive prediction input is missing sidecar text")
        instruction = str(prediction_input.language_instruction).strip()
        if not instruction:
            raise ValueError("Language2Drive sidecar instruction must be non-empty")
        if _alpamayo2_helper is None:
            raise RuntimeError("Alpamayo 2 Super dependencies are unavailable")

        with _ALPAMAYO2_HELPER_LOCK:
            original_create_messages = _alpamayo2_helper.create_messages

            def create_messages(data: dict[str, Any], model_config: Any) -> Any:
                return _alpamayo2_language_messages(
                    data, model_config, instruction
                )

            _alpamayo2_helper.create_messages = create_messages
            try:
                return super().predict(prediction_input)
            finally:
                _alpamayo2_helper.create_messages = original_create_messages
