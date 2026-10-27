from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .data import load_for_config
from .download import download
from .experiment import Experiment, inspect


def main(argv=None):
    parser = argparse.ArgumentParser(description="ContractNLI four-condition base experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="Download and verify the official dataset")
    get.add_argument("--destination", default="data/contract-nli")
    get.add_argument("--archive", help="Use a local official zip (still checksum-verified)")
    freeze = commands.add_parser("freeze", help="Pin cached/remote HF metadata to immutable commits; no weight loading")
    freeze.add_argument("--config", default="configs/rtx5070_v2.toml")
    freeze.add_argument("--destination", default="configs/rtx5070_v2.frozen.json")
    audit = commands.add_parser("audit", help="Read-only legacy/current output validation and paired bootstrap")
    audit.add_argument("--predictions", required=True)
    audit.add_argument("--metrics")
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--bootstrap", type=int, default=20000)
    audit.add_argument("--seed", type=int, default=20261027)
    run = commands.add_parser("run", help="Run or resume an experiment")
    run.add_argument("--config", default="configs/rtx5070_v2.frozen.json")
    run.add_argument("--stage", choices=["all", "summarize", "compress", "judge", "report"], default="all")
    run.add_argument("--max-documents", type=int, help="Smoke-test subset; do not report as full test results")
    run.add_argument("--output-dir")
    run.add_argument("--dry-run", action="store_true", help="Inspect data/count requests; no models, predictions or metrics")
    run.add_argument("--preflight", action="store_true", help="Check full-input context lengths with tokenizers only")
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(args.destination, args.archive))
        return
    if args.command == "freeze":
        from .reproducibility import freeze_config
        config = freeze_config(load_config(args.config), args.destination)
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "audit":
        from .audit import save_audit
        report = save_audit(args.predictions, args.metrics, args.output_dir,
                            bootstrap_samples=args.bootstrap, seed=args.seed)
        print(json.dumps({mode: result["conditions"] for mode, result in report["analyses"].items()},
                         ensure_ascii=False, indent=2))
        return
    if not Path(args.config).is_file():
        parser.error(f"Config not found: {args.config}. Run `python -m contractnli freeze` first.")
    config = load_config(args.config)
    overrides = {}
    if args.max_documents is not None:
        overrides["max_documents"] = args.max_documents
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    config = replace(config, **overrides)
    if args.dry_run:
        print(json.dumps(inspect(config, load_for_config(config)), ensure_ascii=False, indent=2))
        return
    if args.preflight:
        from .preflight import preflight
        result = preflight(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            raise SystemExit(1)
        return
    result = Experiment(config).run(args.stage)
    if result is not None:
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
