#!/usr/bin/env python3
"""reproduce_paper.py — Hua et al. (2024) WarAgent 論文 Table 2-5 ヘッドライン指標 一括再現レポート．

Rust の `waragent reproduce` が書き出す `reproduce_summary.json` (3 トリガー条件
null / dardanelles / archduke の開戦・エスカレーション・同盟分極化と，Table 2-5 の
headline story に対する PASS/off アンカー) を読み，論文の headline 結果を再現する:

  - Table 2 (alliance escalation) : 各トリガー条件の同盟 MI / 紛争数の時系列．
  - Table 3 (cold war)            : 中間強度トリガーで «開戦せず緊張のみ» になる過程．
  - Table 4 (escalation dynamics) : 史実トリガーで同盟国が参戦し紛争対が増える過程．
  - Table 5 (counterfactual)      : トリガー条件別の開戦/冷戦/MI のクロス比較バー．

`--run` を付けると先に Rust バイナリ (`waragent reproduce`) を実行して最新結果を作る．
`--mock` / `--quick` はそのまま Rust バイナリへ渡す (オフライン・短縮再現)．

Usage:
    waragent-tools reproduce
    waragent-tools reproduce --run --mock --quick
    waragent-tools reproduce --results-dir results/20260530_000000_reproduce
    waragent-tools reproduce --json

Outputs:
    <results_dir>/figures/
    ├── table2_alliance_escalation.png   ← トリガー別 同盟 MI / 紛争数 の時系列
    └── table5_trigger_compare.png       ← トリガー別 開戦/冷戦/最終 MI のクロス比較
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from socsim_tools.io import resolve_results_dir

# --------------------------------------------------------------------------- #
# 日本語フォント・カラー設定 (visualize.py と統一)．
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
COLOR_MI = "#2196F3"
COLOR_CONFLICT = "#F44336"
COLOR_OUTBREAK = "#C62828"
COLOR_COLDWAR = "#FF9800"
TRIGGER_COLORS = {
    "null": "#4CAF50",
    "naval-incident": "#03A9F4",
    "dardanelles": "#FF9800",
    "archduke-assassination": "#C62828",
}


def _run_binary(seed: int, mock: bool, quick: bool) -> None:
    """cargo run --release -- reproduce を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed)]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_summary(results_dir: Path) -> dict:
    path = results_dir / "reproduce_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"reproduce_summary.json が見つかりません: {path}\n"
            f"  先に `waragent-tools reproduce --run` を実行してください．"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_metrics(results_dir: Path, subdir: str) -> pd.DataFrame:
    """トリガー条件サブディレクトリの metrics.csv を読む．"""
    return pd.read_csv(results_dir / subdir / "metrics.csv")


def _save_alliance_escalation(
    metrics_by_trigger: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """トリガー別の 同盟 MI / 紛争数 の時系列 (Table 2/4 風)．"""
    fig, (ax_mi, ax_conf) = plt.subplots(
        1, 2, figsize=(13, 5), facecolor=COLOR_BG
    )
    fig.suptitle(
        "Hua et al. (2024) WarAgent — Table 2/4: トリガー別 同盟分極化・エスカレーション",
        fontsize=14,
    )

    for ax in (ax_mi, ax_conf):
        ax.set_facecolor(COLOR_BG)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("ラウンド t")

    for trigger, metrics in metrics_by_trigger.items():
        color = TRIGGER_COLORS.get(trigger, "#777777")
        ax_mi.plot(
            metrics["round"], metrics["alliance_mi"], color=color, lw=2.0, label=trigger
        )
        ax_conf.plot(
            metrics["round"], metrics["n_conflicts"], color=color, lw=2.0, label=trigger
        )

    ax_mi.set_ylabel("同盟分割 MI (vs 史実)")
    ax_mi.set_title("同盟 MI: 史実トリガーほど分極化が進む")
    ax_mi.set_ylim(0.0, 1.0)
    ax_mi.legend(loc="best", fontsize=9)

    ax_conf.set_ylabel("宣戦布告対数 (紛争規模)")
    ax_conf.set_title("紛争規模: 史実トリガーで同盟国が参戦しエスカレーション")
    ax_conf.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _save_trigger_compare(summary: dict, out_path: Path) -> None:
    """トリガー別の 開戦/冷戦/最終 MI のクロス比較バー (Table 3/5 風)．"""
    scenarios = summary.get("scenarios", [])
    triggers = [s["trigger"] for s in scenarios]
    x = range(len(triggers))

    fig, (ax_flag, ax_mi) = plt.subplots(
        1, 2, figsize=(13, 5), facecolor=COLOR_BG
    )
    fig.suptitle(
        "Hua et al. (2024) WarAgent — Table 3/5: トリガー条件別 開戦/冷戦/同盟分極化",
        fontsize=14,
    )

    width = 0.38
    ax_flag.set_facecolor(COLOR_BG)
    ax_flag.bar(
        [i - width / 2 for i in x],
        [s["war_outbreak"] for s in scenarios],
        width=width,
        color=COLOR_OUTBREAK,
        label="開戦 (war_outbreak)",
    )
    ax_flag.bar(
        [i + width / 2 for i in x],
        [s["cold_war_flag"] for s in scenarios],
        width=width,
        color=COLOR_COLDWAR,
        label="冷戦 (cold_war)",
    )
    ax_flag.set_xticks(list(x))
    ax_flag.set_xticklabels(triggers, rotation=20, ha="right", fontsize=8)
    ax_flag.set_ylabel("フラグ (1=該当)")
    ax_flag.set_ylim(0.0, 1.2)
    ax_flag.set_title("null=平時, 中間強度=冷戦, 史実=開戦")
    ax_flag.legend(loc="best", fontsize=9)
    ax_flag.grid(True, axis="y", alpha=0.3)

    ax_mi.set_facecolor(COLOR_BG)
    ax_mi.bar(
        list(x),
        [s["final_alliance_mi"] for s in scenarios],
        color=[TRIGGER_COLORS.get(t, "#777777") for t in triggers],
    )
    ax_mi.set_xticks(list(x))
    ax_mi.set_xticklabels(triggers, rotation=20, ha="right", fontsize=8)
    ax_mi.set_ylabel("最終 同盟分割 MI (vs 史実)")
    ax_mi.set_ylim(0.0, 1.0)
    ax_mi.set_title("史実トリガーほど史実同盟構造へ分極化")
    ax_mi.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _print_table(summary: dict) -> None:
    print("=" * 78)
    print("Hua et al. (2024) WarAgent — Table 2-5 再現レポート")
    print(f"  paper    : {summary.get('paper', '')}")
    print(
        f"  scenario : {summary.get('scenario', '')} | "
        f"mock={summary['mock']} quick={summary['quick']}"
    )
    print("=" * 78)
    for s in summary.get("scenarios", []):
        esc = s["escalation_round"] if s["escalation_round"] >= 0 else "なし"
        print(
            f"  [{s['trigger']:<22}] 開戦={s['war_outbreak']} 冷戦={s['cold_war_flag']} "
            f"勃発R={esc} 紛争={s['n_conflicts']} "
            f"MI={s['final_alliance_mi']:.3f} 宣戦J={s['final_declaration_jaccard']:.3f} "
            f"総動員J={s['final_mobilization_jaccard']:.3f} (round {s['final_round']})"
        )
    print("-" * 78)
    n_pass = 0
    for a in summary.get("anchors", []):
        hi = a["target_hi"]
        hi_str = "∞" if hi is None or hi == float("inf") or hi > 1e30 else f"{hi:.2f}"
        status = "PASS" if a["pass"] else "OFF "
        if a["pass"]:
            n_pass += 1
        print(
            f"[{status}] {a['name']:<52} "
            f"obs={a['observed']:.4f} target=[{a['target_lo']:.2f},{hi_str}] "
            f"paper={a['paper_value']}"
        )
    print("-" * 78)
    print(f"{n_pass}/{len(summary.get('anchors', []))} アンカーが in-band")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="waragent-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=None,
        help="図の保存先 (既定: <results>/figures)",
    )
    parser.add_argument(
        "--run", action="store_true", help="先に Rust バイナリ (waragent reproduce) を実行する．"
    )
    parser.add_argument(
        "--mock", action="store_true", help="--run 時に scripted mock を使う (オフライン)．"
    )
    parser.add_argument(
        "--quick", action="store_true", help="--run 時に短縮再現 (rounds=2)．"
    )
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument(
        "--json", action="store_true", help="サマリを JSON で出力する (図は生成しない)．"
    )
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(args.seed, args.mock, args.quick)

    results_dir = resolve_results_dir(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_table(summary)

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("-" * 78)
    print(f"図の出力先: {out_dir}")

    metrics_by_trigger: dict[str, pd.DataFrame] = {}
    for s in summary.get("scenarios", []):
        try:
            metrics_by_trigger[s["trigger"]] = _load_metrics(
                results_dir, s["results_subdir"]
            )
        except FileNotFoundError:
            continue

    if metrics_by_trigger:
        _save_alliance_escalation(
            metrics_by_trigger, out_dir / "table2_alliance_escalation.png"
        )
    _save_trigger_compare(summary, out_dir / "table5_trigger_compare.png")

    print("-" * 78)
    print("完了．出力ファイル一覧:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s} ({size_kb:6.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
