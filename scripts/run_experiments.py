#!/usr/bin/env python
"""Run the experiments.

    uv run python scripts/run_experiments.py a            # majority baseline
    uv run python scripts/run_experiments.py b            # LLM judge (costs money)
    uv run python scripts/run_experiments.py c            # math -> programming
    uv run python scripts/run_experiments.py d e          # both learning curves
    uv run python scripts/run_experiments.py all          # everything except b
    uv run python scripts/run_experiments.py d --k 8 32 --seeds 0 1

Results land in results/raw/ as one JSON per (experiment, system, domain, k, seed).
Completed runs are skipped unless --force is passed, so an interrupted sweep can
be resumed by rerunning the same command.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import K_VALUES, SEEDS, RunConfig  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("experiments", nargs="+", choices=["a", "b", "c", "d", "e", "all"])
    p.add_argument("--k", type=int, nargs="+", default=list(K_VALUES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--domains", nargs="+", default=["math", "programming"])
    p.add_argument("--config", default=None, help="YAML run config (defaults to configs/base.yaml)")
    p.add_argument("--force", action="store_true", help="rerun even if a result file exists")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="log and skip a failing run instead of stopping the sweep",
    )
    return p.parse_args()


def load_config(path: str | None) -> RunConfig:
    cfg_path = Path(path) if path else CONFIG_DIR / "base.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    return RunConfig.from_yaml(cfg_path)


def fresh_config(base: RunConfig) -> RunConfig:
    """A copy, so one run's mutations don't leak into the next."""
    fields = {k: v for k, v in vars(base).items() if k != "extra"}
    return RunConfig(**fields, extra=dict(base.extra))


def already_done(experiment: str, system: str, domain: str, k: int, seed: int) -> bool:
    from experiments.common import result_path

    return result_path(experiment, system, domain, k, seed).exists()


def main() -> int:
    args = parse_args()
    wanted = set(args.experiments)
    if "all" in wanted:
        # (b) is excluded from 'all' on purpose: it spends API credit.
        wanted = {"a", "c", "d", "e"}

    base_cfg = load_config(args.config)
    failures: list[str] = []

    def guard(label: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {exc}")
            print(f"  ERROR in {label}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                traceback.print_exc()
                raise

    if "a" in wanted:
        from experiments.a_majority import run as run_a

        for domain in args.domains:
            if args.force or not already_done("a", "majority", domain, 0, 0):
                guard(f"a/{domain}", lambda d=domain: run_a(d))

    if "b" in wanted:
        from experiments.b_llm_judge import run as run_b

        for domain in args.domains:
            guard(f"b/{domain}", lambda d=domain: run_b(d))

    if "c" in wanted:
        from experiments.transfer import run_c

        for seed in args.seeds:
            if args.force or not already_done("c", "math_trained", "programming", 0, seed):
                guard(f"c/seed{seed}", lambda s=seed: run_c(s, fresh_config(base_cfg)))

    if "d" in wanted:
        from experiments.transfer import run_d

        for seed in args.seeds:
            for k in args.k:
                if args.force or not already_done("d", "math_plus_k", "programming", k, seed):
                    guard(
                        f"d/k{k}/seed{seed}",
                        lambda k=k, s=seed: run_d(k, s, fresh_config(base_cfg)),
                    )

    if "e" in wanted:
        from experiments.transfer import run_e

        for seed in args.seeds:
            for k in args.k:
                if args.force or not already_done("e", "prog_only", "programming", k, seed):
                    guard(
                        f"e/k{k}/seed{seed}",
                        lambda k=k, s=seed: run_e(k, s, fresh_config(base_cfg)),
                    )

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nDone. Build the table and plot with:")
    print("  uv run python scripts/make_table.py")
    print("  uv run python scripts/make_plots.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
