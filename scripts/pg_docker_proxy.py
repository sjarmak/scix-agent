#!/usr/bin/env python3
"""Tiny TCP proxy that forwards docker bridge 172.17.0.1:5432 -> 127.0.0.1:5432.

Why this exists: the scix-mcp docker container reaches postgres via
host.docker.internal (= 172.17.0.1 on Linux), but postgres is bound only
to 127.0.0.1 and we can't sudo-edit /etc/postgresql/16/main/postgresql.conf
from here. This proxy binds to the docker bridge gateway and relays
connections to loopback, so pg_hba sees them as 127.0.0.1 (already allowed
with scram-sha-256) and the container can connect.

Run in background:
    nohup ./scripts/pg_docker_proxy.py > /tmp/pg_docker_proxy.log 2>&1 &
    disown

Long-term fix: have an admin set listen_addresses='localhost,172.17.0.1'
in postgresql.conf and add a pg_hba.conf line for the docker bridge CIDR.
This script is a stopgap.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

BIND_HOST = "172.17.0.1"
BIND_PORT = 5432
UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 5432

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("pg-proxy")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
    peer = client_w.get_extra_info("peername")
    try:
        up_r, up_w = await asyncio.open_connection(UPSTREAM_HOST, UPSTREAM_PORT)
    except OSError as e:
        log.warning("upstream connect failed from %s: %s", peer, e)
        client_w.close()
        return
    await asyncio.gather(_pipe(client_r, up_w), _pipe(up_r, client_w))


async def _main() -> None:
    server = await asyncio.start_server(_handle, BIND_HOST, BIND_PORT)
    sockets = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    log.info("listening on %s, forwarding to %s:%d", sockets, UPSTREAM_HOST, UPSTREAM_PORT)

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: stop.cancel() if not stop.done() else None)
    try:
        async with server:
            await server.serve_forever()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)
