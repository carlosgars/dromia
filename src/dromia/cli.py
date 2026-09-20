"""Single DromIA command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dromia import config as dromia_config
from dromia import stack
from dromia.models import store
from dromia.pipeline import runner


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="dromia")
    commands = root.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="install or verify external model artifacts")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    install = model_commands.add_parser("install")
    install.add_argument("--accept-licenses", action="store_true")
    model_commands.add_parser("verify")
    run = commands.add_parser("run", help="analyze one video")
    run.add_argument("video", type=Path)
    run.add_argument("--capture-fps", type=float)
    run.add_argument("--runs-dir", type=Path)
    stack_parser = commands.add_parser("stack", help="manage the local DromIA/CVAT stack")
    stack_parser.add_argument("action", choices=("up", "down", "health"))
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "models":
        result = (
            store.install(accept_licenses=args.accept_licenses)
            if args.models_command == "install"
            else store.verify()
        )
    elif args.command == "run":
        base = dromia_config.DromiaConfig()
        updates: dict[str, object] = {}
        if args.runs_dir is not None:
            updates["runs_dir"] = args.runs_dir
        updates["gait_analysis"] = base.gait_analysis.model_copy(
            update={"capture_fps_override": args.capture_fps}
        )
        result = runner.run(args.video, base.model_copy(update=updates)).model_dump(mode="json")
    else:
        result = (
            stack.up()
            if args.action == "up"
            else stack.down()
            if args.action == "down"
            else stack.health()
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
