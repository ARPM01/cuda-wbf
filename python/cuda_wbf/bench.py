"""CPU/GPU fixture and COCO verification."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

from .verification import verify_cpu, verify_cuda


def main(argv=None):
    parser = argparse.ArgumentParser(description="CUDA WBF verification and benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="check backends and record the benchmark environment")
    preflight_parser.add_argument("--cpu-only", action="store_true", help="check only the scalar backend")
    verify = commands.add_parser("verify", help="check native fusion against the pinned Python oracle")
    verify.add_argument("--cpu-only", action="store_true", help="verify with the CPU backend")
    verify.add_argument("--fixtures-only", action="store_true", help="run named fixtures without randomized cases")
    verify.add_argument("--predictions", type=Path, help="upstream benchmark.zip")
    verify.add_argument("--annotations", type=Path, help="COCO instances_val2017.json")
    verify.add_argument("--baselines", action="store_true", help="evaluate NMS comparisons on the COCO inputs")
    verify.add_argument("--output", type=Path, help="write quality.json and metadata.json to this directory")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        from .preflight import preflight

        try:
            print(json.dumps(preflight(cpu_only=args.cpu_only), indent=2))
        except (ValueError, RuntimeError, OSError) as error:
            print(f"Preflight failed: {error}", file=sys.stderr)
            return 1
        return 0
    if (args.predictions is None) != (args.annotations is None):
        verify.error("--predictions and --annotations must be supplied together")
    if args.predictions is None and (args.baselines or args.output is not None):
        verify.error("--baselines and --output require --predictions and --annotations")
    try:
        if args.cpu_only:
            report = verify_cpu(True) if args.fixtures_only else verify_cpu()
        else:
            report = verify_cuda(fixtures_only=args.fixtures_only)
        if args.predictions is not None:
            from .coco_quality import verify_coco

            quality, metadata = verify_coco(args.predictions, args.annotations, report, baselines=args.baselines)
            report = {**report, "quality": quality, "metadata": metadata}
            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=True)
                for name in ("quality", "metadata"):
                    (args.output / f"{name}.json").write_text(json.dumps(report[name], indent=2) + "\n",
                                                            encoding="utf-8")
    except (AssertionError, ValueError, RuntimeError, OSError, BadZipFile) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
