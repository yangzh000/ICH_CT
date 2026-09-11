import argparse
from pathlib import Path

import pandas as pd

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hemorrhage")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
        if name == "run":
            command.add_argument("--resume", action="store_true")
    demo = commands.add_parser("demo")
    demo.add_argument("--config", required=True)
    demo.add_argument("--output", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    from .matlab_bridge import close_matlab_services
    try:
        if args.command == "predict":
            from .workflow import predict_saved
            predict_saved(args.model, args.manifest, args.output)
        else:
            config = load_config(args.config)
            if args.command == "demo":
                from .demo import generate_demo
                generate_demo(args.output, config)
            elif args.command == "evaluate":
                from .evaluation import evaluate_external
                evaluate_external(pd.read_csv(args.predictions, dtype={"patient_id": str}), args.output, config["evaluation"])
            elif args.command == "run":
                from .workflow import run_study
                run_study(args.manifest, config, args.output, args.resume)
            else:
                from .common import read_manifest, write_json
                from .imaging import CaseStore
                from .workflow import validate_cohorts
                patients = read_manifest(args.manifest)
                development, external = validate_cohorts(patients, config)
                store = CaseStore(Path(args.output) / "cache", config)
                store.validate_unique_images(patients)
                write_json(Path(args.output) / "input_validation.json", {"status": "passed", "development_n": len(development), "external_n": len(external), "source_hashes": store.hashes})
        print(f"Completed: {Path(args.output).resolve()}", flush=True)
    finally:
        close_matlab_services()
