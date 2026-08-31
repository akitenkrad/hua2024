#!/usr/bin/env python3
"""reproduce_paper.py — Hua et al. (2024) WarAgent 論文 Table 2-5 ヘッドライン指標 一括再現レポート．

`waragent reproduce` の親 run とその子 run を読み，論文の headline 結果を再現する:

  - Table 2 (alliance escalation) : 各トリガー条件の同盟 MI / 紛争数の時系列．
  - Table 3 (cold war)            : 中間強度トリガーで «開戦せず緊張のみ» になる過程．
  - Table 4 (escalation dynamics) : 史実トリガーで同盟国が参戦し紛争対が増える過程．
  - Table 5 (counterfactual)      : トリガー条件別の開戦/冷戦/MI のクロス比較バー．

トリガー条件ごとの値は子 run (`metrics.csv`) が，条件をまたいだ同盟分極化のギャップ
は親の `scope=sweep` 指標が持つ (旧 `reproduce_summary.json` は書かない — 同じ値が
run ディレクトリの中にある)．

アンカーの帯と PASS/OFF はここでは出さない．帯は論文の主張ではなくこの再現実装が
置いたものなので Rust 側のコンソールにだけ残す — 同じ閾値を Python と Rust の
2 箇所に置くと食い違う余地ができる．

`--run` を付けると先に Rust バイナリ (`waragent reproduce`) を実行して最新結果を作る．
`--mock` / `--quick` はそのまま Rust バイナリへ渡す (オフライン・短縮再現)．

Usage:
    waragent-tools reproduce
    waragent-tools reproduce --run --mock --quick
    waragent-tools reproduce --results-dir "$(runvault path --experiment waragent --latest --subcommand reproduce)"
    waragent-tools reproduce --json

Outputs:
    results/waragent/figures/<run_slug>/
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
from runvault.read import config_parameters, figures_dir, sweep_children

from waragent_tools.runs import load_metrics, resolve_run_dir, run_scope

# --------------------------------------------------------------------------- #
# 日本語フォント・カラー設定 (visualize.py と統一)．
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
COLOR_OUTBREAK = "#C62828"
COLOR_COLDWAR = "#FF9800"
TRIGGER_COLORS = {
    "null": "#4CAF50",
    "naval-incident": "#03A9F4",
    "dardanelles": "#FF9800",
    "archduke-assassination": "#C62828",
}

# 論文が示す «トリガー強度の順序» に合わせた表示順．
TRIGGER_ORDER = ["null", "naval-incident", "dardanelles", "archduke-assassination"]


def _run_binary(seed: int, mock: bool, quick: bool) -> None:
    """cargo run --release -- reproduce を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed)]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_conditions(parent_dir: str) -> list[dict]:
    """トリガー条件ごとの観測を子 run から組み直す (旧 `scenarios` に対応)．"""
    rows: list[dict] = []
    for child in sweep_children(parent_dir):
        params = config_parameters(child) or {}
        scope = run_scope(child)
        steps = load_metrics(child)
        last = steps.iloc[-1]
        rows.append(
            {
                "trigger": params.get("trigger"),
                "war_outbreak": int(last["war_outbreak"]),
                "cold_war_flag": int(scope["cold_war_flag"]),
                # 勃発しなかった run には行そのものが無い (0 で埋めない)．
                "escalation_round": scope.get("escalation_round"),
                "n_conflicts": int(last["n_conflicts"]),
                "final_alliance_mi": float(last["alliance_mi"]),
                "final_declaration_jaccard": float(last["declaration_jaccard"]),
                "final_mobilization_jaccard": float(last["mobilization_jaccard"]),
                "final_round": int(scope["final_round"]),
                "run_dir": child,
                "metrics": steps,
            }
        )
    if not rows:
        raise SystemExit(
            f"エラー: この reproduce 親に子 run がありません: {parent_dir}\n"
            "  先に `waragent-tools reproduce --run --mock --quick` を実行してください．"
        )
    order = {t: i for i, t in enumerate(TRIGGER_ORDER)}
    rows.sort(key=lambda r: order.get(r["trigger"], len(order)))
    return rows


def _save_alliance_escalation(conditions: list[dict], out_path: Path) -> None:
    """トリガー別の 同盟 MI / 紛争数 の時系列 (Table 2/4 風)．"""
    fig, (ax_mi, ax_conf) = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Hua et al. (2024) WarAgent — Table 2/4: トリガー別 同盟分極化・エスカレーション",
        fontsize=14,
    )

    for ax in (ax_mi, ax_conf):
        ax.set_facecolor(COLOR_BG)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("ラウンド t")

    for c in conditions:
        color = TRIGGER_COLORS.get(c["trigger"], "#777777")
        metrics: pd.DataFrame = c["metrics"]
        ax_mi.plot(
            metrics["round"], metrics["alliance_mi"], color=color, lw=2.0, label=c["trigger"]
        )
        ax_conf.plot(
            metrics["round"], metrics["n_conflicts"], color=color, lw=2.0, label=c["trigger"]
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


def _save_trigger_compare(conditions: list[dict], out_path: Path) -> None:
    """トリガー別の 開戦/冷戦/最終 MI のクロス比較バー (Table 3/5 風)．"""
    triggers = [c["trigger"] for c in conditions]
    x = range(len(triggers))

    fig, (ax_flag, ax_mi) = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Hua et al. (2024) WarAgent — Table 3/5: トリガー条件別 開戦/冷戦/同盟分極化",
        fontsize=14,
    )

    width = 0.38
    ax_flag.set_facecolor(COLOR_BG)
    ax_flag.bar(
        [i - width / 2 for i in x],
        [c["war_outbreak"] for c in conditions],
        width=width,
        color=COLOR_OUTBREAK,
        label="開戦 (war_outbreak)",
    )
    ax_flag.bar(
        [i + width / 2 for i in x],
        [c["cold_war_flag"] for c in conditions],
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
        [c["final_alliance_mi"] for c in conditions],
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


def _print_table(params: dict, conditions: list[dict], sweep: dict) -> None:
    print("=" * 78)
    print("Hua et al. (2024) WarAgent — Table 2-5 再現レポート")
    print(
        f"  scenario : {params.get('scenario', '')} | "
        f"mock={params.get('mock')} rounds={params.get('rounds')}"
    )
    print("=" * 78)
    for c in conditions:
        esc = "なし" if c["escalation_round"] is None else int(c["escalation_round"])
        print(
            f"  [{c['trigger']:<22}] 開戦={c['war_outbreak']} 冷戦={c['cold_war_flag']} "
            f"勃発R={esc} 紛争={c['n_conflicts']} "
            f"MI={c['final_alliance_mi']:.3f} 宣戦J={c['final_declaration_jaccard']:.3f} "
            f"総動員J={c['final_mobilization_jaccard']:.3f} (round {c['final_round']})"
        )
    print("-" * 78)
    gap = sweep.get("alliance_mi_gap_archduke_minus_null")
    if gap is not None:
        print(f"  同盟分極化のギャップ (archduke − null): {gap:+.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="waragent-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results root (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=None,
        help="図の保存先 (既定: results/waragent/figures/<run_slug>/)",
    )
    parser.add_argument(
        "--run", action="store_true", help="先に Rust バイナリ (waragent reproduce) を実行する．"
    )
    parser.add_argument(
        "--mock", action="store_true", help="--run 時に scripted mock を使う (オフライン)．"
    )
    parser.add_argument("--quick", action="store_true", help="--run 時に短縮再現 (rounds=2)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument(
        "--json", action="store_true", help="サマリを JSON で出力する (図は生成しない)．"
    )
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(args.seed, args.mock, args.quick)

    parent = resolve_run_dir(
        args.results_dir, args.results_root, subcommand="reproduce", standalone=False
    )
    params = config_parameters(parent) or {}
    sweep = run_scope(parent)
    conditions = load_conditions(parent)

    if args.json:
        payload = {
            "parent": parent,
            "parameters": params,
            "sweep_metrics": sweep,
            "conditions": [
                {k: v for k, v in c.items() if k != "metrics"} for c in conditions
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    _print_table(params, conditions, sweep)

    out_dir = Path(args.output_dir) if args.output_dir else Path(figures_dir(parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("-" * 78)
    print(f"図の出力先: {out_dir}")

    _save_alliance_escalation(conditions, out_dir / "table2_alliance_escalation.png")
    _save_trigger_compare(conditions, out_dir / "table5_trigger_compare.png")

    print("-" * 78)
    print("完了．出力ファイル一覧:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s} ({size_kb:6.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
