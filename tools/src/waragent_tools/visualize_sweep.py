#!/usr/bin/env python3
"""
visualize_sweep.py — Hua et al. (2024) WarAgent スイープ結果 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv を読み，
トリガー強度 × スタンス の格子について，
(1) 開戦発生率ヒートマップ (trigger × stance),
(2) 同盟構造類似度 (alliance_mi) ヒートマップ,
(3) トリガー別の開戦率・冷戦率の棒グラフ
を生成する (微小トリガーでも冷戦/開戦に至る傾向の確認)．

Usage:
    uv run waragent-tools visualize-sweep
    uv run waragent-tools visualize-sweep --sweep_dir results/20260524_160000_sweep

Outputs:
    output_dir/
    ├── sweep_outbreak_heatmap.png   ← 開戦率 (trigger × stance)
    ├── sweep_alliance_mi_heatmap.png ← 同盟 MI (trigger × stance)
    └── sweep_trigger_bars.png       ← トリガー別 開戦率 / 冷戦率
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"


def load_summary(sweep_dir: str) -> pd.DataFrame:
    path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    # トリガーラベル "null" は pandas が NaN と誤解釈するため，trigger 列だけ
    # 既定 NA 変換を無効化して文字列のまま読む (escalation_round 等の空欄 NaN は維持)．
    df = pd.read_csv(path)
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    for col in ("trigger", "stance", "scenario"):
        if col in raw.columns:
            df[col] = raw[col]
    return df


def _heatmap(df: pd.DataFrame, value: str, agg: str, title: str, out_path: str, vmax: float = 1.0) -> None:
    """trigger × stance の格子で `value` を `agg` 集約してヒートマップ化する．"""
    table = df.pivot_table(index="trigger", columns="stance", values=value, aggfunc=agg)
    fig, ax = plt.subplots(
        figsize=(1.8 + 1.6 * table.shape[1], 1.4 + 0.9 * table.shape[0]),
        facecolor=COLOR_BG,
    )
    ax.set_facecolor(COLOR_BG)
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="magma", aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=20, ha="right")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("スタンス")
    ax.set_ylabel("トリガー")
    ax.set_title(title, fontsize=12)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_trigger_bars(df: pd.DataFrame, out_path: str) -> None:
    """トリガー別の開戦率・冷戦率を棒グラフで比較する (微小トリガー → 冷戦の確認)．"""
    triggers = sorted(df["trigger"].unique())
    outbreak = [df[df["trigger"] == t]["war_outbreak"].mean() * 100.0 for t in triggers]
    coldwar = [df[df["trigger"] == t]["cold_war_flag"].mean() * 100.0 for t in triggers]

    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(triggers), 4.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    x = np.arange(len(triggers))
    w = 0.38
    ax.bar(x - w / 2, outbreak, w, color="#F44336", alpha=0.85, label="開戦率")
    ax.bar(x + w / 2, coldwar, w, color="#2196F3", alpha=0.85, label="冷戦率")
    for i, v in enumerate(outbreak):
        ax.text(i - w / 2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(coldwar):
        ax.text(i + w / 2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(triggers, rotation=20, ha="right")
    ax.set_ylabel("発生率 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("トリガー別 開戦率 / 冷戦率 (強度↑ → 開戦↑)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="waragent-tools visualize-sweep",
        description="Hua et al. (2024) WarAgent スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        default="results/latest",
        help="スイープ出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Hua et al. (2024) WarAgent スイープ可視化 ===")
    print(f"スイープ: {args.sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/3] sweep_summary.csv を読み込み中 ...")
    df = load_summary(args.sweep_dir)
    print(
        f"      トリガー {df['trigger'].nunique()} 種 × スタンス {df['stance'].nunique()} 種 "
        f"(計 {len(df)} 実行)"
    )

    print("[2/3] 開戦率 / 同盟 MI ヒートマップを保存中 ...")
    _heatmap(df, "war_outbreak", "mean", "開戦率 (trigger × stance)",
             os.path.join(out_dir, "sweep_outbreak_heatmap.png"))
    _heatmap(df, "final_alliance_mi", "mean", "同盟 MI (trigger × stance)",
             os.path.join(out_dir, "sweep_alliance_mi_heatmap.png"))

    print("[3/3] トリガー別 開戦率 / 冷戦率 棒グラフを保存中 ...")
    save_trigger_bars(df, os.path.join(out_dir, "sweep_trigger_bars.png"))

    print("-------------------------------------------------")
    print("トリガー別の開戦発生率 / 平均 同盟MI:")
    for t in sorted(df["trigger"].unique()):
        sub = df[df["trigger"] == t]
        freq = sub["war_outbreak"].mean() * 100.0
        mi = sub["final_alliance_mi"].mean()
        cw = sub["cold_war_flag"].mean() * 100.0
        print(f"  trigger={t} → 開戦 = {freq:.1f}% | 冷戦 = {cw:.1f}% | 同盟MI = {mi:.3f}")

    print("-------------------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
