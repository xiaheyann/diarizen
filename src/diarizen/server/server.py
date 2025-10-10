from __future__ import annotations

import logging

import grpc
from google.protobuf.duration_pb2 import Duration

from diarizen.inferencer.grpc_inferencer import GrpcInferencer
from diarizen.proto import ux_speaker_diarization_pb2 as pb2
from diarizen.proto import ux_speaker_diarization_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


# ----------------------------- Servicer -----------------------------------
class UxSpeakerDiarizationServicer(pb2_grpc.UxSpeakerDiarizationServicer):
    def __init__(self):
        logger.info("Initializing UxSpeakerDiarizationServicer")
        self.inferencer = GrpcInferencer()
        logger.info("UxSpeakerDiarizationServicer initialized successfully")

    def Detect(self, request: pb2.DetectRequest, context: grpc.ServicerContext) -> pb2.DetectResponse:  # type: ignore[override]
        logger.info("Received Detect request")
        # 解析grpc request为python常用类型
        audio = request.audio
        sample_rate = request.config.sample_rate_hertz
        encoding = request.config.encoding
        logger.info(f"Audio size: {len(audio)} bytes, sample_rate: {sample_rate}, encoding: {encoding}")
        
        # 推理
        try:
            results = self.inferencer.detect(audio, sample_rate, encoding)
            logger.info(f"Detection completed, found {len(results) if results else 0} segments")
        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)
            raise
        
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

        logger.info(f"Returning response with {len(pb_results)} results")
        return pb2.DetectResponse(results=pb_results)

    def DetectWav(self, request: pb2.DetectWavRequest, context: grpc.ServicerContext) -> pb2.DetectResponse:  # type: ignore[override]
        path = request.path
        sample_rate = request.config.sample_rate_hertz
        encoding = request.config.encoding
        logger.info(f"Received DetectWav request for path: {path}, sample_rate: {sample_rate}, encoding: {encoding}")
        
        try:
            results = self.inferencer.detect_wav(path, sample_rate, encoding, context)
            logger.info(f"DetectWav completed, found {len(results) if results else 0} segments")
        except Exception as e:
            logger.error(f"Error during DetectWav: {e}", exc_info=True)
            raise
        
        pb_results = []
        for r in results or []:
            pb_result = pb2.DetectionResult(
                start_time=self._to_duration(r.get("start_time", 0)),
                end_time=self._to_duration(r.get("end_time", 0)),
                speaker=r.get("speaker", 0),
            )
            pb_results.append(pb_result)
        
        logger.info(f"Returning DetectWav response with {len(pb_results)} results")
        return pb2.DetectResponse(results=pb_results)

    @staticmethod
    def _to_duration(seconds: float):
        d = Duration()
        # 把秒和小数部分分开
        d.seconds = int(seconds)  # 取整秒
        d.nanos = int((seconds - d.seconds) * 1e9)  # 转换成纳秒
        return d
