from __future__ import annotations

import logging
from concurrent import futures

import grpc

from diarizen.proto import ux_speaker_diarization_pb2 as pb2
from diarizen.proto import ux_speaker_diarization_pb2_grpc as pb2_grpc


# ----------------------------- Servicer -----------------------------------
class UxSpeakerDiarizationServicer(pb2_grpc.UxSpeakerDiarizationServicer):
    def Detect(self, request: pb2.DetectRequest, context: grpc.ServicerContext) -> pb2.DetectResponse:  # type: ignore[override]
        pass

    def DetectWav(self, request: pb2.DetectWavRequest, context: grpc.ServicerContext) -> pb2.DetectResponse:  # type: ignore[override]
        pass


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
