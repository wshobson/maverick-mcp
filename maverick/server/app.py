"""CLI entry point: parses --transport and starts the assembled server. Top layer: imports only assembly."""

import argparse
import sys

from maverick.platform.telemetry import get_logger, setup_logging
from maverick.server.assembly import build_server

_DEFAULT_HTTP_HOST = "127.0.0.1"
# No server-port field exists on `platform.config.PlatformSettings` today
# (it only covers database/redis/cache/http-client/telemetry), so this
# falls back to the legacy server's own default port.
_DEFAULT_HTTP_PORT = 8003


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="maverick.server",
        description="maverick-mcp: personal-use MCP server for stock analysis.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="MCP transport: stdio (default, for Claude Desktop) or streamable HTTP.",
    )
    parser.add_argument(
        "--host",
        default=_DEFAULT_HTTP_HOST,
        help=f"HTTP host, --transport http only (default: {_DEFAULT_HTTP_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_HTTP_PORT,
        help=f"HTTP port, --transport http only (default: {_DEFAULT_HTTP_PORT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args, build the server, and run it on the requested transport.

    A failure building the server (invalid settings, a bad `DATABASE_URL`,
    etc.) is caught here and reported as a clean one-line error on stderr
    plus a non-zero exit, rather than a raw traceback -- this is the
    process's only top-level entry point, so this is the only place that
    can offer that.
    """
    args = _parse_args(argv)
    setup_logging()
    logger = get_logger("maverick.server")

    try:
        mcp = build_server()
    except Exception as exc:
        logger.error("maverick.server: failed to start: %s", exc, exc_info=True)
        print(f"maverick-mcp: failed to start: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
