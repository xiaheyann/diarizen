import json
import math
from pathlib import Path
from typing import Callable, Optional, Text, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio
from scipy.ndimage import median_filter
from tqdm import tqdm

from diarizen.inferencer.diarizen.powerset import Powerset


class Inferencer:

    @classmethod
    def from_json(
        cls,
        diarizen_hub: Path,
        device: Optional[torch.device] = torch.device("cuda"),
    ):
        config = json.load((diarizen_hub / "config.json").open("r", encoding="utf-8"))
        inference_config = config["inference"]["args"]
        model = ort.InferenceSession(
            str(diarizen_hub / "onnx_model.onnx"), providers=["CPUExecutionProvider"]
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
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        sr: int = 16000,
        max_speakers_per_chunk: int = 4,
    ):
        self.model = model
        self.device = device
        self.sr = sr
        self.window = window
        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.duration = duration
        # ~~~~ powerset to multilabel conversion ~~~~
        self.skip_conversion = skip_conversion
        self.conversion = Powerset(self.max_speakers_per_chunk, 2).to(self.device)
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

    def model_forward(
        self, chunks: torch.Tensor, soft=False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        # Extract model inputs from the chunks tensor
        inputs = {self.model.get_inputs()[0].name: chunks.cpu().numpy()}
        # Run inference with ONNX Runtime
        outputs = self.model.run(None, inputs)[0]
        outputs = torch.from_numpy(outputs).to(self.device)
        return self.conversion(outputs, soft=soft).cpu().numpy()

    def slide(
        self,
        waveform: torch.Tensor,
        soft: bool = False,
    ) -> np.ndarray:
        sample_rate = self.sr
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
        bar = tqdm(
            np.arange(0, num_chunks, self.batch_size),
            desc="Processing chunks",
            leave=False,
        )
        for c in bar:
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
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        outputs = self.slide(waveform, soft=False)
        # 中值滤波
        outputs = median_filter(outputs, size=(1, 11, 1), mode="reflect")
        return outputs
