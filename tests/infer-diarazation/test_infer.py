import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from usieg.formatter import audition as aufmt

from diarizen.inferencer.diarizen.inferencer import Inferencer


def infer(inferencer: Inferencer, wav_path: Path, output_dir: Path):
    waveform, sr = librosa.load(
        wav_path,
        sr=None,
        mono=True,
        dtype=np.float32,
    )
    waveform = waveform[-16 * sr :]
    diar_results = inferencer.infer(waveform, sr)
    sf.write(output_dir / "16s.wav", waveform, sr)
    result = diar_results.swapaxes(1, 2)[0]
    results = []
    for idx, trace in enumerate(result):
        df = aufmt.mask_to_au_df(trace, 50)
        if len(df) > 0:
            df["Name"] = f"Speaker-{idx + 1}"
            results.append(df)
    if len(results) > 0:
        df = pd.concat(results, ignore_index=True)
        df.to_csv(output_dir / "result03.csv", index=False, sep="\t")


if __name__ == "__main__":
    wav_path = Path("data-bin/qhd-test3/test02.wav")
    output_dir = Path("data-bin/qhd-test3")
    output_dir.mkdir(exist_ok=True, parents=True)
    diar_pipeline = Inferencer.from_json(
        Path("model-bin/diarizen-wavlm-base"),
        device="cpu",
    )
    # infer(diar_pipeline, wav_path, output_dir)
    infer(diar_pipeline, wav_path, output_dir)

