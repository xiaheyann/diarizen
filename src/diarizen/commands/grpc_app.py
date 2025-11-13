import logging
from concurrent import futures

import grpc
import typer

app = typer.Typer()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(50051, help="Server port"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    max_workers: int = typer.Option(10, help="Maximum number of worker threads"),
    max_message_size: int = typer.Option(
        100 * 1024 * 1024, help="Maximum gRPC message size in bytes"
    ),
) -> None:

    from diarizen.proto import ux_speaker_diarization_pb2 as pb2
    from diarizen.proto import ux_speaker_diarization_pb2_grpc as pb2_grpc
    from diarizen.server.server import UxSpeakerDiarizationServicer

    options = [
        ("grpc.max_receive_message_length", max_message_size),
        ("grpc.max_send_message_length", max_message_size),
    ]
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=options
    )
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


def main():
    app()


if __name__ == "__main__":
    app()
