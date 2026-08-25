"""Entry point: `headroom` or `python -m headroom`.

Loopback only, and not configurable to a routable address from the command line.
Exposing an interface that can start and stop processes to the network is not a
setting anyone should be one typo away from.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .app import Settings, create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="headroom", description=__doc__)
    parser.add_argument("--port", type=int, default=7315, help="port for the Headroom UI itself")
    parser.add_argument(
        "--server-port", type=int, default=8080, help="port llama-server listens on"
    )
    parser.add_argument("--registry", help="path to models.json")
    parser.add_argument("--llama-server", help="path to the llama-server executable")
    parser.add_argument("--reload", action="store_true", help="auto-reload for development")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = Settings()
    if args.registry:
        settings.registry_path = type(settings.registry_path)(args.registry)
    if args.llama_server:
        settings.llama_server = type(settings.llama_server)(args.llama_server)
    settings.port = args.server_port

    app = create_app(settings)
    print(f"\n  Headroom -> http://127.0.0.1:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
