#!/usr/bin/env python
"""Compile the Upstox Market Data Feed V3 protobuf schema.

The V3 market feed is protobuf-encoded. This script downloads the OFFICIAL
schema from Upstox and compiles it locally, which is the only correct way to
decode the feed - guessing field numbers would silently corrupt market data.

Requires:  pip install grpcio-tools

If you do not run this, the system still works: it reports
DECODER_UNAVAILABLE and uses the REST market-quote feed instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

PROTO_URL = "https://assets.upstox.com/feed/market-data-feed/v3/MarketDataFeed.proto"


def main() -> int:
    out_dir = REPO_ROOT / "var" / "cache" / "proto"
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_path = out_dir / "MarketDataFeed.proto"

    print(f"Downloading {PROTO_URL}")
    try:
        import httpx

        response = httpx.get(PROTO_URL, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        print(f"Could not download the schema: {exc}")
        print("The REST market-quote feed will be used instead (no schema required).")
        return 1

    proto_path.write_text(response.text, encoding="utf-8")
    print(f"Saved {proto_path}")

    try:
        import grpc_tools  # noqa: F401
    except ImportError:
        print()
        print("grpcio-tools is not installed, so the schema cannot be compiled.")
        print("  pip install grpcio-tools")
        print("  python scripts/download_proto.py")
        print()
        print("Until then the system uses the REST feed. That is safe and correct - it")
        print("just means the websocket stream is unavailable.")
        return 1

    print("Compiling ...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{out_dir}",
            f"--python_out={out_dir}",
            str(proto_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        return 1
    compiled = out_dir / "marketdatafeed_pb2.py"
    if compiled.exists():
        print(f"Compiled {compiled}")
        print("Restart the app and the V3 websocket feed will be used automatically.")
        return 0
    print("Compilation produced no module; check the proto file.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
