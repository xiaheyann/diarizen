import grpc
import numpy as np
import soundfile as sf

from diarizen.proto import ux_speaker_diarization_pb2 as pb2
from diarizen.proto import ux_speaker_diarization_pb2_grpc as pb2_grpc


def read_wav_as_pcm_bytes(wav_path):
    data, sample_rate = sf.read(wav_path, dtype="int16")
    # 保证单通道
    if data.ndim == 2:
        assert data.shape[1] == 1, "只支持单通道wav"
        data = data[:, 0]
    pcm_bytes = data.tobytes()
    return pcm_bytes, sample_rate


def main():
    wav_path = "data-bin/qhd-test3/test02.wav"
    pcm_bytes, sample_rate = read_wav_as_pcm_bytes(wav_path)
    # 构造Config
    config = pb2.Config(encoding=pb2.Config.LINEAR16, sample_rate_hertz=sample_rate)
    # 构造DetectRequest
    request = pb2.DetectRequest(config=config, audio=pcm_bytes)
    # gRPC请求
    channel = grpc.insecure_channel("localhost:50051")
    stub = pb2_grpc.UxSpeakerDiarizationStub(channel)
    response = stub.Detect(request)
    print("DetectResponse:")
    for r in response.results:
        print(
            f"speaker={r.speaker}, start={r.start_time.seconds + r.start_time.nanos/1e9:.2f}s, end={r.end_time.seconds + r.end_time.nanos/1e9:.2f}s"
        )


if __name__ == "__main__":
    main()
