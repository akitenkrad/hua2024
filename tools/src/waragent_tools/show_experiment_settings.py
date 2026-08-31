"""waragent-tools show-experiment-settings — 実行結果の設定表示．

runvault の run ディレクトリの `config.json` (封筒．条件は `parameters` の下) を
読み，実行時に使われた全パラメータを整形表示する．run / sweep / reproduce の
どれかは `run.json` の `subcommand` が答える．LLM の同一性 (provider・モデル・
温度) は `run.json` の `llm` ブロック，呼び出し数と cache-hit・開戦・冷戦などの
結果は `metrics.csv` の step を持たない run スコープ行が持つ (旧
`run_metadata.json` は書かない — 同じ値が run ディレクトリの中にある)．

legacy な出力 (`results/<timestamp>/` の flat な `config.json` と
`run_metadata.json`，`results/<timestamp>_sweep/sweep_config.json`) も
`--results-dir` に直接渡せば従来どおり読める．

run ディレクトリのパスは次で取れる:
    runvault path --experiment waragent --latest --subcommand run --standalone
    runvault path --experiment waragent --latest --subcommand sweep

Usage:
    waragent-tools show-experiment-settings
    waragent-tools show-experiment-settings --results-dir "$(runvault path --experiment waragent --latest --subcommand run --standalone)"
    waragent-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runvault.read import config_parameters, load_run_meta

from waragent_tools.runs import resolve_run_dir, run_scope

# run の config キー → 表示ラベル (従来と同じ桁揃え)．
RUN_FIELDS = [
    ("scenario", "シナリオ         "),
    ("trigger", "トリガー         "),
    ("stance_override", "スタンス上書き   "),
    ("secretary_passes", "秘書検証パス     "),
    ("rounds", "ラウンド数       "),
    ("war_threshold", "開戦しきい値     "),
    ("n_countries", "国数             "),
    ("seed", "シード (コア)    "),
    ("llm_temperature", "LLM 温度         "),
    ("llm_seed", "LLM seed         "),
]

# run スコープ指標 → 表示ラベル．
METRIC_FIELDS = [
    ("llm_calls", "呼び出し総数     "),
    ("llm_cache_hits", "cache-hit        "),
    ("final_round", "実行ラウンド数   "),
    ("cold_war_flag", "冷戦フラグ       "),
    ("escalation_round", "勃発ラウンド     "),
]

DETERMINISM_NOTE = (
    "LLM output is outside socsim bit-reproducibility; the prompt->response cache "
    "(with temperature=0 and fixed seed) is the reproducibility mechanism. The socsim "
    "core (scenario/board init, activation order, publicity propagation, alliance/war "
    "resolution, escalation, board updates and all metrics) is deterministic given the "
    "seed. LLM calls per round = n_countries * (1 + secretary_passes)."
)


def _load(results_dir: Path) -> tuple[dict, Path, str]:
    """実験条件と，それがどのサブコマンドのものかを返す．

    runvault の `config.json` は封筒で，条件は `parameters` の下にある．legacy の
    flat な `config.json` は `command` を持ち，legacy の掃引は `sweep_config.json`
    に条件を書いていた．
    """
    params = config_parameters(results_dir, required=False)
    if params is not None:
        meta = load_run_meta(results_dir, required=False)
        if meta is not None:
            kind = str(meta.get("subcommand", "run"))
        else:
            kind = "sweep" if params.get("command") == "sweep" else "run"
        return params, results_dir / "config.json", kind

    sweep_cfg = results_dir / "sweep_config.json"
    if sweep_cfg.exists():
        with sweep_cfg.open() as f:
            return json.load(f), sweep_cfg, "sweep"

    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (runvault の封筒 / legacy の flat) "
        f"または sweep_config.json (legacy の sweep)"
    )


def render_run_config(cfg: dict, source: Path) -> str:
    lines = ["=" * 70, "実行設定 (run)", "=" * 70, f"設定ファイル: {source}", "-" * 70]
    for key, label in RUN_FIELDS:
        lines.append(f"{label}: {cfg.get(key, '-')}")
    # 出力先は run ディレクトリそのものなので条件には含まれない (legacy のみ持つ)．
    if cfg.get("output_dir") is not None:
        lines.append(f"出力先           : {cfg['output_dir']}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path, kind: str) -> str:
    """掃引 (sweep / reproduce) 親の設定テーブル．リスト項目は `, ` 連結する．"""
    lines = [
        "=" * 70,
        f"実行設定 ({kind})",
        "=" * 70,
        f"設定ファイル: {source}",
        "-" * 70,
        f"シナリオ         : {cfg.get('scenario', '-')}",
        f"トリガー候補     : {', '.join(map(str, cfg.get('trigger_values', [])))}",
    ]
    if "stance_values" in cfg:
        lines.append(f"スタンス候補     : {', '.join(map(str, cfg['stance_values']))}")
    lines.append(f"秘書検証パス     : {cfg.get('secretary_passes', '-')}")
    lines.append(f"ラウンド数       : {cfg.get('rounds', '-')}")
    lines.append(f"開戦しきい値     : {cfg.get('war_threshold', '-')}")
    if "runs" in cfg:
        lines.append(f"試行数 runs      : {cfg['runs']}")
    if "mock" in cfg:
        lines.append(f"mock             : {cfg['mock']}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm(meta: dict | None, scope: dict, legacy: dict | None) -> str:
    """LLM の同一性 (run.json の llm ブロック) と実行の結果 (run スコープ指標)．"""
    llm = (meta or {}).get("llm") or {}
    lines = ["", "LLM 実行メタデータ (run.json の llm ブロック / run スコープ指標)", "-" * 70]
    if legacy is not None:
        # legacy の run_metadata.json は endpoint も持っていた．
        lines.append(f"モデル           : {legacy.get('llm_model', '-')}")
        lines.append(f"endpoint         : {legacy.get('llm_endpoint', '-')}")
        lines.append(f"温度             : {legacy.get('llm_temperature', '-')}")
        lines.append(f"seed             : {legacy.get('llm_seed', '-')}")
        lines.append(f"呼び出し総数     : {legacy.get('total_calls', '-')}")
        lines.append(f"cache-hit        : {legacy.get('cache_hits', '-')}")
        rate = legacy.get("cache_hit_rate")
        if rate is not None:
            lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
        lines.append(f"開戦             : {legacy.get('war_outbreak', '-')}")
        lines.append(f"勃発ラウンド     : {legacy.get('escalation_round', '-')}")
        lines.append(f"紛争数           : {legacy.get('n_conflicts', '-')}")
        lines.append(f"冷戦フラグ       : {legacy.get('cold_war_flag', '-')}")
    else:
        lines.append(f"provider         : {llm.get('provider', '-')}")
        lines.append(f"モデル           : {llm.get('model_snapshot', '-')}")
        lines.append(f"温度             : {llm.get('temperature', '-')}")
        for key, label in METRIC_FIELDS:
            value = scope.get(key)
            lines.append(f"{label}: {'-' if value is None else value}")
        rate = scope.get("llm_cache_hit_rate")
        # 呼び出しが 1 本も無い run には率の行そのものが無い (0 で埋めない)．
        if rate is not None:
            lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    lines.append("-" * 70)
    lines.append(f"注記: {DETERMINISM_NOTE}")
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
        default=None,
        help="run ディレクトリ (省略時は runvault path --latest が解決する)",
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results root (default: results)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = Path(resolve_run_dir(args.results_dir, args.results_root))
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg, cfg_path, kind = _load(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    meta = load_run_meta(results_dir, required=False)
    # legacy の wide な metrics.csv には run スコープ行が無い (空の辞書が返る)．
    # legacy は同じ値を run_metadata.json が持っているので，そちらを読む．
    scope = run_scope(str(results_dir))
    legacy_meta_path = results_dir / "run_metadata.json"
    legacy = None
    if meta is None and legacy_meta_path.exists():
        with legacy_meta_path.open() as f:
            legacy = json.load(f)

    if args.json:
        payload = {
            "source": str(cfg_path),
            "kind": kind,
            "config": cfg,
            "llm": (meta or {}).get("llm"),
            "run_metrics": scope,
        }
        if legacy is not None:
            payload["run_metadata"] = legacy
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if kind == "run":
        print(render_run_config(cfg, cfg_path))
        print(render_llm(meta, scope, legacy))
    else:
        print(render_sweep_config(cfg, cfg_path, kind))
    return 0


if __name__ == "__main__":
    sys.exit(main())
