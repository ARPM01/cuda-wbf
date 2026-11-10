"""CPU/GPU fixture verification; data and performance commands follow later."""

import argparse
import json
import sys

from .verification import verify_cpu, verify_cuda


def main(argv=None):
    parser = argparse.ArgumentParser(description="CUDA WBF verification and benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="check native fusion against the pinned Python oracle")
    verify.add_argument("--cpu-only", action="store_true", help="run CPU fixtures without COCO assets or CUDA")
    verify.add_argument("--fixtures-only", action="store_true", help="run named fixtures without randomized cases")
    args = parser.parse_args(argv)
    try:
        if args.cpu_only:
            report = verify_cpu(True) if args.fixtures_only else verify_cpu()
        else:
            report = verify_cuda(fixtures_only=args.fixtures_only)
    except (AssertionError, ValueError, RuntimeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
