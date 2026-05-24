"""waragent-tools show-experiment-settings — 実行結果の設定表示．

results/{timestamp}/config.json (run) または
results/{timestamp}_sweep/sweep_config.json (sweep) を読み，実行時に使われた全
パラメータを整形表示する．存在すれば run_metadata.json の LLM 情報
(モデル・endpoint・温度・seed・cache-hit 率・開戦・冷戦) も併せて表示する．
`results/latest` も解決される．

Usage:
    waragent-tools show-experiment-settings
    waragent-tools show-experiment-settings --results-dir results/20260524_153000
    waragent-tools show-experiment-settings --results-dir results/latest --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_results_dir(arg: str) -> Path:
    """ユーザ指定の results_dir を絶対パスに解決する (symlink も実体へ)．"""
    p = Path(arg)
    if not p.is_absolute():
        candidates = [Path.cwd() / arg, p]
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            p = candidates[0]
    return Path(os.path.realpath(p))


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """config.json (run) か sweep_config.json (sweep) を探す．"""
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (run) または sweep_config.json (sweep)"
    )


def _load_run_metadata(results_dir: Path) -> dict | None:
    path = results_dir / "run_metadata.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def render_run_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (run)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"シナリオ         : {cfg.get('scenario', '-')}")
    lines.append(f"トリガー         : {cfg.get('trigger', '-')}")
    lines.append(f"スタンス上書き   : {cfg.get('stance_override', '-')}")
    lines.append(f"秘書検証パス     : {cfg.get('secretary_passes', '-')}")
    lines.append(f"ラウンド数       : {cfg.get('rounds', '-')}")
    lines.append(f"開戦しきい値     : {cfg.get('war_threshold', '-')}")
    lines.append(f"国数             : {cfg.get('n_countries', '-')}")
    lines.append(f"シード (コア)    : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append(f"出力先           : {cfg.get('output_dir', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"シナリオ         : {cfg.get('scenario', '-')}")
    lines.append(f"トリガー候補     : {', '.join(map(str, cfg.get('trigger_values', [])))}")
    lines.append(f"スタンス候補     : {', '.join(map(str, cfg.get('stance_values', [])))}")
    lines.append(f"秘書検証パス     : {cfg.get('secretary_passes', '-')}")
    lines.append(f"ラウンド数       : {cfg.get('rounds', '-')}")
    lines.append(f"開戦しきい値     : {cfg.get('war_threshold', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_run_metadata(meta: dict) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (run_metadata.json)")
    lines.append("-" * 70)
    lines.append(f"モデル           : {meta.get('llm_model', '-')}")
    lines.append(f"endpoint         : {meta.get('llm_endpoint', '-')}")
    lines.append(f"温度             : {meta.get('llm_temperature', '-')}")
    lines.append(f"seed             : {meta.get('llm_seed', '-')}")
    lines.append(f"呼び出し総数     : {meta.get('total_calls', '-')}")
    lines.append(f"cache-hit        : {meta.get('cache_hits', '-')}")
    rate = meta.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    lines.append(f"開戦             : {meta.get('war_outbreak', '-')}")
    lines.append(f"勃発ラウンド     : {meta.get('escalation_round', '-')}")
    lines.append(f"紛争数           : {meta.get('n_conflicts', '-')}")
    lines.append(f"冷戦フラグ       : {meta.get('cold_war_flag', '-')}")
    note = meta.get("determinism_note")
    if note:
        lines.append("-" * 70)
        lines.append(f"注記: {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="waragent-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="実行結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = _resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    cfg_path, kind = _find_config_file(results_dir)
    with cfg_path.open() as f:
        cfg = json.load(f)
    meta = _load_run_metadata(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "run_metadata": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_run_metadata(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
