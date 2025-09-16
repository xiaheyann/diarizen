from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diarizen.proto import ux_speaker_diarization_pb2 as pb2

from typing import Any, Dict, Optional

import numpy as np


class GrpcInferencer:

    def __init__(self):
        pass

    def detect(
        self,
        audio: bytes,
        sample_rate: int,
        encoding: int,
    ) -> list:
        """
        audio: 原始音频数据（bytes）
        sample_rate: 采样率
        encoding: 编码类型（int, 见proto定义）
        context: 可选，grpc context
        return: list of dict, 每个dict包含start_time, end_time, speaker
        """
        audio_np = np.frombuffer(audio, dtype=np.int16)
        pass

    def detect_wav(
        self,
        path: str,
        sample_rate: int,
        encoding: int,
        context: Optional[Any] = None,
    ) -> list:
        """
        path: wav文件路径
        sample_rate: 采样率
        encoding: 编码类型（int, 见proto定义）
        context: 可选，grpc context
        return: list of dict, 每个dict包含start_time, end_time, speaker
        """
        pass
