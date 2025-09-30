from __future__ import annotations

import grpc
from google.protobuf.duration_pb2 import Duration

from diarizen.inferencer.grpc_inferencer import GrpcInferencer
from diarizen.proto import ux_speaker_diarization_pb2 as pb2
from diarizen.proto import ux_speaker_diarization_pb2_grpc as pb2_grpc


# ----------------------------- Servicer -----------------------------------
class UxSpeakerDiarizationServicer(pb2_grpc.UxSpeakerDiarizationServicer):
    def __init__(self):
        self.inferencer = GrpcInferencer()

    def Detect(self, request: pb2.DetectRequest, context: grpc.ServicerContext) -> pb2.DetectResponse:  # type: ignore[override]
        # 解析grpc request为python常用类型
        audio = request.audio
        sample_rate = request.config.sample_rate_hertz
        encoding = request.config.encoding
        # 推理
        results = self.inferencer.detect(audio, sample_rate, encoding)
        # results: list of dict, 每个dict包含start_time(float秒), end_time(float秒), speaker(int)
        # 构造pb2.DetectResponse
        pb_results = []
        for r in results or []:
            pb_result = pb2.DetectionResult(
                start_time=self._to_duration(r.get("start_time", 0)),
                end_time=self._to_duration(r.get("end_time", 0)),
                speaker=r.get("speaker", 0),
            )
            pb_results.append(pb_result)
        print("DetectResponse:")
        for r in pb_results:
            print(
                f"speaker={r.speaker}, start={r.start_time.seconds + r.start_time.nanos/1e9:.2f}s, end={r.end_time.seconds + r.end_time.nanos/1e9:.2f}s"
            )

        return pb2.DetectResponse(results=pb_results)

    def DetectWav(self, request: pb2.DetectWavRequest, context: grpc.ServicerContext) -> pb2.DetectResponse:  # type: ignore[override]
        path = request.path
        sample_rate = request.config.sample_rate_hertz
        encoding = request.config.encoding
        results = self.inferencer.detect_wav(path, sample_rate, encoding, context)
        pb_results = []
        for r in results or []:
            pb_result = pb2.DetectionResult(
                start_time=self._to_duration(r.get("start_time", 0)),
                end_time=self._to_duration(r.get("end_time", 0)),
                speaker=r.get("speaker", 0),
            )
            pb_results.append(pb_result)
        return pb2.DetectResponse(results=pb_results)

    @staticmethod
    def _to_duration(seconds: float):
        d = Duration()
        # 把秒和小数部分分开
        d.seconds = int(seconds)  # 取整秒
        d.nanos = int((seconds - d.seconds) * 1e9)  # 转换成纳秒
        return d
