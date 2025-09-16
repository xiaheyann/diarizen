from __future__ import annotations

import logging
from concurrent import futures

import grpc

from diarizen.proto import ux_speaker_diarization_pb2 as pb2
from diarizen.proto import ux_speaker_diarization_pb2_grpc as pb2_grpc
from .grpc_inferencer import GrpcInferencer



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
        results = self.inferencer.detect(audio, sample_rate, encoding, context)
        # results: list of dict, 每个dict包含start_time(float秒), end_time(float秒), speaker(int)
        # 构造pb2.DetectResponse
        pb_results = []
        for r in results or []:
            pb_result = pb2.DetectionResult(
                start_time=self._to_duration(r.get('start_time', 0)),
                end_time=self._to_duration(r.get('end_time', 0)),
                speaker=r.get('speaker', 0),
            )
            pb_results.append(pb_result)
        return pb2.DetectResponse(results=pb_results)

    def DetectWav(self, request: pb2.DetectWavRequest, context: grpc.ServicerContext) -> pb2.DetectResponse:  # type: ignore[override]
        path = request.path
        sample_rate = request.config.sample_rate_hertz
        encoding = request.config.encoding
        results = self.inferencer.detect_wav(path, sample_rate, encoding, context)
        pb_results = []
        for r in results or []:
            pb_result = pb2.DetectionResult(
                start_time=self._to_duration(r.get('start_time', 0)),
                end_time=self._to_duration(r.get('end_time', 0)),
                speaker=r.get('speaker', 0),
            )
            pb_results.append(pb_result)
        return pb2.DetectResponse(results=pb_results)

    @staticmethod
    def _to_duration(seconds: float):
        # 转为google.protobuf.duration_pb2.Duration
        from google.protobuf.duration_pb2 import Duration
        d = Duration()
        d.FromSeconds(seconds)
        return d


# ------------------------------- Server -----------------------------------


def serve(host: str, port: int, log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_UxSpeakerDiarizationServicer_to_server(
        UxSpeakerDiarizationServicer(), server
    )
    server.add_insecure_port(f"{host}:{port}")
    logging.info("Starting UxSpeakerDiarization server on %s:%s", host, port)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        server.stop(grace=None)


if __name__ == "__main__":
    # -------------------- Hardcoded settings here --------------------
    HOST = "0.0.0.0"
    PORT = 50051
    LOG_LEVEL = "INFO"
    # ----------------------------------------------------------------
    serve(HOST, PORT, LOG_LEVEL)
