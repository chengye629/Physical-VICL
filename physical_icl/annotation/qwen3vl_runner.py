#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _local_files_only() -> bool:
    return os.environ.get("PHYSICL_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no"}


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_python(x) for x in value]
    if isinstance(value, list):
        return [_to_python(x) for x in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


class VideoRunner:
    """Qwen3-VL video inference wrapper used by Physics Card Pass A."""

    def __init__(self, model_path: Path | str) -> None:
        model_ref = str(model_path)
        local_only = _local_files_only()
        self.processor = AutoProcessor.from_pretrained(model_ref, local_files_only=local_only)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=local_only,
        )
        self.model.eval()

    def prepare(self, messages: list[dict[str, Any]]):
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        patch_size = getattr(self.processor.image_processor, "patch_size", 16)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadata = None
        sampling: dict[str, Any] = {}
        if video_inputs is not None:
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadata = list(video_metadata)
            if video_inputs:
                sampling["actual_sampled_frames"] = int(video_inputs[0].shape[0])

        fps_value = video_kwargs.get("fps")
        if isinstance(fps_value, (list, tuple)) and len(fps_value) == 1:
            video_kwargs["fps"] = float(fps_value[0])

        sampling["video_kwargs"] = _to_python(video_kwargs)
        sampling["video_metadata"] = _to_python(video_metadata)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        return inputs.to(self.model.device), sampling

    @torch.inference_mode()
    def generate(
        self,
        video_path: Path,
        prompt: str,
        fps: float,
        min_frames: int,
        max_frames: int,
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path.resolve()}",
                    "fps": fps,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                    "min_pixels": 128 * 128,
                    "max_pixels": 448 * 448,
                    "total_pixels": 64 * 448 * 448,
                },
                {"type": "text", "text": prompt},
            ],
        }]
        inputs, sampling = self.prepare(messages)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return text, sampling


class TextRunner:
    """Qwen3-VL text-only inference wrapper used by Pass B."""

    def __init__(self, model_path: Path | str) -> None:
        model_ref = str(model_path)
        local_only = _local_files_only()
        self.processor = AutoProcessor.from_pretrained(model_ref, local_files_only=local_only)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_ref,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=local_only,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
