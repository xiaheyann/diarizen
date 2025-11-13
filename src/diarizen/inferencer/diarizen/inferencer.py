import json
import logging
import math
from pathlib import Path
from typing import Callable, Optional, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.ndimage import median_filter

from diarizen.inferencer.diarizen.powerset import Powerset
from diarizen.models.diarizen.model import Model as EENDModel

logger = logging.getLogger(__name__)


class Inferencer:

    @classmethod
    def from_json(cls, diarizen_hub: Path, device: str = "cpu"):
        config = json.load((diarizen_hub / "config.json").open("r", encoding="utf-8"))
        inference_config = config["inference"]["args"]
        model = EENDModel(**config["model"]["args"])
        model.load_state_dict(
            torch.load(diarizen_hub / "pytorch_model.bin", map_location=device)
        )
        seg_duration = inference_config["seg_duration"]

        return cls(
            model,
            duration=seg_duration,
            step=inference_config["segmentation_step"] * seg_duration,
            skip_aggregation=True,
            batch_size=inference_config["batch_size"],
            device=device,
        )

    def __init__(
        self,
        model,
        window: Text = "sliding",
        duration: Optional[float] = None,
        step: Optional[float] = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        skip_conversion: bool = False,
        device: str = "cpu",
        batch_size: int = 32,
        auto_clear_cache: bool = True,
    ):
        device = torch.device(device)
        
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.auto_clear_cache = auto_clear_cache

        self.window = window

        duration = duration
        self.duration = duration
        # ~~~~ powerset to multilabel conversion ~~~~
        self.skip_conversion = skip_conversion
        self.conversion = Powerset(self.model.max_speakers_per_chunk, 2).to(self.device)
        # ~~~~ overlap-add aggregation ~~~~~
        self.skip_aggregation = skip_aggregation
        self.pre_aggregation_hook = pre_aggregation_hook

        # step between consecutive chunks
        if step > self.duration:
            raise ValueError(
                f"Step between consecutive chunks is set to {step:g}s, while chunks are "
                f"only {self.duration:g}s long, leading to gaps between consecutive chunks. "
                f"Either decrease step or increase duration."
            )
        self.step = step
        self.batch_size = batch_size

        logger.info(
            f"Inferencer initialized on device: {self.device}, auto_clear_cache: {auto_clear_cache}"
        )

    def model_forward(
        self, chunks: torch.Tensor, soft=False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        with torch.inference_mode():
            chunks_gpu = chunks.to(self.device)
            outputs = self.model(chunks_gpu)
            # 转换并立即移到CPU，释放GPU上的中间结果
            result = self.conversion(outputs, soft=soft).cpu().numpy()
            # 删除GPU上的张量引用
            del chunks_gpu, outputs
        return result

    def slide(
        self,
        waveform: torch.Tensor,
        soft: bool = False,
    ) -> np.ndarray:
        sample_rate = self.model.sample_rate
        window_size = math.floor(self.duration * sample_rate)
        step_size = round(self.step * sample_rate)
        _, num_samples = waveform.shape
        # prepare complete chunks
        if num_samples >= window_size:
            # "channel chunk frame -> chunk channel frame"
            chunks = waveform.unfold(1, window_size, step_size).permute(1, 0, 2)
            num_chunks = chunks.shape[0]
        else:
            num_chunks = 0
        # prepare last incomplete chunk
        has_last_chunk = (num_samples < window_size) or (
            num_samples - window_size
        ) % step_size > 0
        if has_last_chunk:
            # pad last chunk with zeros
            last_chunk = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = F.pad(last_chunk, (0, last_pad))
        outputs = []
        # 分块遍历
        for c in np.arange(0, num_chunks, self.batch_size):
            batch = chunks[c : c + self.batch_size]
            batch_outputs = self.model_forward(batch, soft=soft)
            outputs.append(batch_outputs)
        if has_last_chunk:
            last_outputs = self.model_forward(last_chunk[None], soft=soft)
            outputs.append(last_outputs)
        outputs = np.vstack(outputs)
        return outputs

    def infer(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        waveform = torch.from_numpy(waveform).unsqueeze(0)  # shape: (1, num_samples)
        if sr != self.model.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.model.sample_rate
            )

        try:
            outputs = self.slide(waveform, soft=False)
            # 中值滤波
            outputs = median_filter(outputs, size=(1, 11, 1), mode="reflect")
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA显存不足！推理过程中GPU显存已耗尽。错误详情: {e}")
            # 尝试清理显存后重新抛出异常
            self.clear_cache()
            raise
        except Exception as e:
            logger.error(f"推理过程中发生错误: {e}", exc_info=True)
            raise
        finally:
            # 清理GPU缓存，释放未使用的显存
            self.clear_cache()

        return outputs

    def clear_cache(self):
        """清理GPU缓存，释放未使用的显存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if self.device.type == "npu":
            torch.npu.empty_cache()
            torch.npu.synchronize()
