"""Strict buffered reader for pinned AlpaSim length-delimited protobuf logs."""

import struct


def iter_asl(path, max_message_bytes=128 * 1024 * 1024):
    from alpasim_grpc.v0.logging_pb2 import LogEntry

    with open(path, "rb", buffering=1024 * 1024) as stream:
        while True:
            prefix = stream.read(4)
            if not prefix:
                return
            if len(prefix) != 4:
                raise ValueError("truncated ASL message length")
            size, = struct.unpack(">L", prefix)
            if size < 1 or size > max_message_bytes:
                raise ValueError("invalid or oversized ASL message")
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError("truncated ASL message payload")
            message = LogEntry.FromString(payload)
            if message.WhichOneof("log_entry") is None:
                raise ValueError("ASL message has no recognized entry")
            yield message
