from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diarizen.proto import ux_speaker_diarization_pb2 as pb2

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from diarizen.inferencer.diarizen.inferencer import Inferencer
from diarizen.utils import audition_formatter as aufmt

DOWNSAMPLED_SR = 50

logger = logging.getLogger(__name__)

try:
    torch.npu.set_compile_mode(jit_compile=False)
    logger.info("NPU jit compile mode set to False.")
except AttributeError:
    logger.info("NPU acceleration is unavailable.")


class GrpcInferencer:

    def __init__(self, device: str = "cpu"):
        self.inferencer = Inferencer.from_json(
            Path("model-bin/diarizen-wavlm-base"), device=device
        )
        self.inferencer.step = 16
        self.request_count = 0
        logger.info(
            f"GrpcInferencer initialized successfully on device: {device}, step: {self.inferencer.step}s)"
        )

    def detect(self, audio: bytes, sr: int, encoding: int) -> List[Dict]:
        """
        audio: 原始音频数据（bytes）
        sample_rate: 采样率
        encoding: 编码类型（int, 见proto定义）
        context: 可选，grpc context
        return: list of dict, 每个dict包含start_time, end_time, speaker
        """
        self.request_count += 1
        logger.info(
            f"[Request {self.request_count}] Starting detection - audio_size: {len(audio)} bytes, sample_rate: {sr}, encoding: {encoding}"
        )

        outputs = None
        try:
            audio = np.frombuffer(audio, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0  # 转为float32，归一化到[-1, 1]

            logger.info(f"[Request {self.request_count}] Running inference...")
            outputs = self.inferencer.infer(audio, sr)  # (窗口数, 时间步数, 4)

            outputs = outputs.swapaxes(1, 2)  # (窗口数, 4, 时间步数)
            results = []
            start_base = 0
            for context_idx, context in enumerate(outputs):
                for speaker_idx, trace in enumerate(context):
                    starts, durations = aufmt.mask_to_seconds(trace, DOWNSAMPLED_SR)
                    starts = starts + start_base
                    if len(starts) > 0:
                        df = pd.DataFrame(
                            columns=["context_idx", "speaker_idx", "start", "duration"]
                        )
                        df["start"] = starts
                        df["duration"] = durations
                        df["context_idx"] = context_idx
                        df["speaker_idx"] = speaker_idx
                        df["end"] = df["start"] + df["duration"]
                        df["name"] = context_idx * 10 + speaker_idx
                        results.append(df)
                start_base += self.inferencer.step

            # 没有说话人，直接返回
            if len(results) == 0:
                logger.info(f"[Request {self.request_count}] No speakers detected")
                return []

            df = pd.concat(results, ignore_index=True)
            df = df.sort_values(by=["context_idx", "start", "speaker_idx"])
            audf = aufmt.seconds_to_au_df(
                starts=df["start"].values, durations=df["duration"].values
            )
            audf["Name"] = df["name"].values

            results = []
            for _, row in df.iterrows():
                results.append(
                    {
                        "start_time": float(row["start"]),
                        "end_time": float(row["end"]),
                        "speaker": int(row["name"]),
                    }
                )

            logger.info(
                f"[Request {self.request_count}] Detection completed successfully - found {len(results)} segments from {df['name'].nunique()} speakers"
            )
            return results
        except Exception as e:
            logger.error(
                f"[Request {self.request_count}] Error during detection: {e}",
                exc_info=True,
            )
            raise
        finally:
            # 确保每次请求后都清理显存
            if "audio" in locals():
                del audio
            if outputs is not None:
                del outputs
            if hasattr(self.inferencer, "clear_cache"):
                self.inferencer.clear_cache()

    def detect_wav(self, path: str, sample_rate: int, encoding: int) -> List:
        """
        path: wav文件路径
        sample_rate: 采样率
        encoding: 编码类型（int, 见proto定义）
        context: 可选，grpc context
        return: list of dict, 每个dict包含start_time, end_time, speaker
        """
        logger.warning(
            f"detect_wav called but not implemented - path: {path}, sample_rate: {sample_rate}, encoding: {encoding}"
        )
        pass
