"""Command-line entry point: ``ur-mcp`` / ``python -m ur_mcp``."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ur-mcp", description="MCP server: UR3e / UR5e probe positioning on PCB test points")
    p.add_argument("--transport", choices=["stdio", "http", "sse"], default="stdio")
    p.add_argument("--host", default="127.0.0.1", help="HTTP bind address (not the robot)")
    p.add_argument("--port", type=int, default=10941)
    p.add_argument("--robot", help="robot IP / hostname to auto-connect (sets UR_HOST); 'SIM' for the simulator")
    p.add_argument("--simulate", action="store_true", help="shorthand for --robot SIM")
    p.add_argument("--log-level", default=os.environ.get("UR_LOG_LEVEL", "WARNING"))
    args = p.parse_args(argv)

    # stdout is the MCP channel in stdio mode - log to stderr only.
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.simulate:
        os.environ["UR_HOST"] = "SIM"
    elif args.robot:
        os.environ["UR_HOST"] = args.robot
    if os.environ.get("UR_BOARD_FILE"):
        logging.getLogger("ur_mcp").warning(
            "UR_BOARD_FILE is no longer used: load the board explicitly with robot_board(operation='load').")

    from .server import mcp

    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port, show_banner=False)


if __name__ == "__main__":
    main()
