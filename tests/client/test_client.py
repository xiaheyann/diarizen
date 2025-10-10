import grpc
import numpy as np
import soundfile as sf
from pathlib import Path  # 使用 Path 替代 os

from diarizen.proto import ux_speaker_diarization_pb2 as pb2
from diarizen.proto import ux_speaker_diarization_pb2_grpc as pb2_grpc


def read_wav_as_pcm_bytes(wav_path):
    data, sample_rate = sf.read(wav_path, dtype="int16")
    # 保证单通道
    if data.ndim == 2:
        assert data.shape[1] == 1, "只支持单通道wav"
        data = data[:, 0]
    pcm_bytes = data.tobytes()
    return pcm_bytes, sample_rate, data  # 返回原始波形数据以供切片


def save_segments(results, wav_data, sample_rate, base_path: Path):
    base_path.mkdir(parents=True, exist_ok=True)
    for idx, r in enumerate(results, 1):
        speaker = r.speaker or "speaker"
        start_time = r.start_time.seconds + r.start_time.nanos / 1e9
        end_time = r.end_time.seconds + r.end_time.nanos / 1e9
        start_idx = max(0, int(round(start_time * sample_rate)))
        end_idx = min(len(wav_data), int(round(end_time * sample_rate)))
        if end_idx <= start_idx:
            continue
        segment = wav_data[start_idx:end_idx]
        out_path = base_path / f"{speaker}_{start_time:.2f}_{end_time:.2f}.wav"
        sf.write(str(out_path), segment, sample_rate, subtype="PCM_16")
        print(f"保存音频片段: {out_path}")


def main():
    wav_path = "data-bin/swk/random_concat_hardcoded.wav"
    pcm_bytes, sample_rate, wav_data = read_wav_as_pcm_bytes(wav_path)
    # 构造Config
    config = pb2.Config(encoding=pb2.Config.LINEAR16, sample_rate_hertz=sample_rate)
    # 构造DetectRequest
    request = pb2.DetectRequest(config=config, audio=pcm_bytes)
    # gRPC请求
    channel = grpc.insecure_channel("localhost:51003")
    stub = pb2_grpc.UxSpeakerDiarizationStub(channel)
    response = stub.Detect(request)
    print("DetectResponse:")
    for r in response.results:
        print(
            f"speaker={r.speaker}, start={r.start_time.seconds + r.start_time.nanos/1e9:.2f}s, end={r.end_time.seconds + r.end_time.nanos/1e9:.2f}s"
        )
    # 按时间戳切分波形到 data-bin/swk-trim，使用说话人名称命名
    save_segments(response.results, wav_data, sample_rate, Path("data-bin/swk-trim"))


if __name__ == "__main__":
    main()
