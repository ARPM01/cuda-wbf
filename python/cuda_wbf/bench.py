"""CPU verification entry point; performance commands follow in milestone 3."""

import argparse
import json
import sys

from .verification import verify_cpu


def main(argv=None):
    parser = argparse.ArgumentParser(description="CUDA WBF verification and benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="check native fusion against the pinned Python oracle")
    verify.add_argument("--cpu-only", action="store_true", help="run CPU fixtures without COCO assets or CUDA")
    args = parser.parse_args(argv)
    if not args.cpu_only:
        parser.error("this milestone supports fixture verification only; pass verify --cpu-only")
    try:
        report = verify_cpu()
    except (AssertionError, ValueError, RuntimeError) as error:
        print(f"CPU verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
