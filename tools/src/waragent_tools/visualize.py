#!/usr/bin/env python3
"""
visualize.py — Hua et al. (2024) WarAgent シミュレーション 可視化スクリプト

results/latest (または --results_dir 指定先) の metrics.csv (round, alliance_mi, ...) と
events.csv (round, actor, action, target, publicity) を読み，
(1) 同盟ネットワーク図 (最終ラウンドの同盟 M エッジを networkx で描画),
(2) Board 関係遷移 (ラウンドごとの宣戦布告/同盟/総動員の本数),
(3) 指標時系列 (alliance_mi / declaration_jaccard / mobilization_jaccard),
(4) 紛争規模・総動員数の時系列 (n_conflicts / n_mobilized)
の 4 図 (2×2) を生成する．

Usage:
    uv run waragent-tools visualize
    uv run waragent-tools visualize --results_dir results/20260524_153000
    uv run waragent-tools visualize --output_dir out

Outputs:
    output_dir/
    └── war_dynamics.png   ← 同盟ネットワーク・Board 遷移・指標時系列・紛争規模
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_MI = "#2196F3"
COLOR_DECL = "#F44336"
COLOR_MOB = "#FF9800"
COLOR_ALLIANCE = "#4CAF50"
COLOR_WAR = "#F44336"


def _letter(idx: int) -> str:
    """raw u64 id を匿名国名ラベルへ (0 -> A)．"""
    return f"Country {chr(ord('A') + int(idx))}"


def load_metrics(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    return pd.read_csv(path)


def load_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # events.csv は任意 (古い run には無いかもしれない)．
        return pd.DataFrame(columns=["round", "actor", "action", "target", "publicity"])
    return pd.read_csv(path)


def _build_relation_graph(events: pd.DataFrame, up_to_round: int) -> nx.Graph:
    """events から指定ラウンドまでの «最終» 同盟/宣戦エッジを再構成する．

    各 (actor, target) 対について最後の関係を採用する (alliance / declare_war /
    peace で上書き)．escalate_join_war も war として扱う．
    """
    rel: dict[tuple[int, int], str] = {}
    nodes: set[int] = set()
    sub = events[events["round"] <= up_to_round]
    for _, row in sub.iterrows():
        actor = int(row["actor"])
        nodes.add(actor)
        action = str(row["action"])
        tgt = row["target"]
        if pd.isna(tgt):
            continue
        target = int(tgt)
        nodes.add(target)
        key = (min(actor, target), max(actor, target))
        if action in ("alliance",):
            rel[key] = "alliance"
        elif action in ("declare_war", "escalate_join_war"):
            rel[key] = "war"
        elif action == "peace":
            rel.pop(key, None)

    g = nx.Graph()
    for n in sorted(nodes):
        g.add_node(n)
    for (a, b), kind in rel.items():
        g.add_edge(a, b, kind=kind)
    return g


def save_war_dynamics(metrics: pd.DataFrame, events: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=COLOR_BG)
    fig.suptitle("Hua et al. (2024) WarAgent — 世界大戦外交動態", fontsize=14)

    rounds = sorted(metrics["round"].unique())
    last_round = max(rounds) if rounds else 0

    # --- (0,0) 同盟/宣戦ネットワーク (最終ラウンド) ---
    ax = axes[0, 0]
    ax.set_facecolor(COLOR_BG)
    g = _build_relation_graph(events, last_round)
    if g.number_of_nodes() == 0:
        # events が無い場合はメトリクスのクラスタ数だけ注記する．
        ax.text(0.5, 0.5, "イベント情報なし", ha="center", va="center")
        ax.axis("off")
    else:
        pos = nx.circular_layout(g)
        alliance_edges = [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == "alliance"]
        war_edges = [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == "war"]
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#BBDEFB", node_size=900, edgecolors="#333")
        nx.draw_networkx_labels(
            g, pos, ax=ax, labels={n: _letter(n) for n in g.nodes()}, font_size=8
        )
        nx.draw_networkx_edges(
            g, pos, ax=ax, edgelist=alliance_edges, edge_color=COLOR_ALLIANCE, width=2.5, label="同盟 M"
        )
        nx.draw_networkx_edges(
            g, pos, ax=ax, edgelist=war_edges, edge_color=COLOR_WAR, width=2.5, style="dashed",
            label="宣戦 W"
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"同盟 (緑) / 宣戦 (赤破線) ネットワーク (round {last_round})")

    # --- (0,1) Board 遷移 (ラウンドごとの行動本数) ---
    ax = axes[0, 1]
    ax.set_facecolor(COLOR_BG)
    if not events.empty:
        counts = (
            events[events["action"].isin(["declare_war", "alliance", "non_aggression", "mobilize"])]
            .groupby(["round", "action"])
            .size()
            .unstack(fill_value=0)
        )
        for action, color in [
            ("declare_war", COLOR_WAR),
            ("alliance", COLOR_ALLIANCE),
            ("non_aggression", COLOR_MI),
            ("mobilize", COLOR_MOB),
        ]:
            if action in counts.columns:
                ax.plot(counts.index, counts[action].values, marker="o", ms=4, color=color, label=action)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "イベント情報なし", ha="center", va="center")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("行動本数")
    ax.set_title("Board 遷移 (行動種別ごとの本数)")
    ax.grid(True, alpha=0.3)

    # --- (1,0) 指標時系列 (MI / Jaccard) ---
    ax = axes[1, 0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(metrics["round"], metrics["alliance_mi"], marker="o", ms=4, color=COLOR_MI,
            label="alliance_mi")
    ax.plot(metrics["round"], metrics["declaration_jaccard"], marker="s", ms=4, color=COLOR_DECL,
            label="declaration_jaccard")
    ax.plot(metrics["round"], metrics["mobilization_jaccard"], marker="^", ms=4, color=COLOR_MOB,
            label="mobilization_jaccard")
    ax.set_ylim(0, 1)
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("スコア (vs 史実)")
    ax.set_title("指標時系列 (同盟 MI / 宣戦・総動員 Jaccard)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,1) 紛争規模・総動員数 ---
    ax = axes[1, 1]
    ax.set_facecolor(COLOR_BG)
    ax.plot(metrics["round"], metrics["n_conflicts"], marker="o", ms=4, color=COLOR_WAR,
            label="n_conflicts (宣戦対)")
    ax.plot(metrics["round"], metrics["n_mobilized"], marker="s", ms=4, color=COLOR_MOB,
            label="n_mobilized (総動員国)")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("本数 / 国数")
    ax.set_title("紛争規模・総動員数の推移 (エスカレーション)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="waragent-tools visualize",
        description="Hua et al. (2024) WarAgent 外交動態 可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="Rust シミュレーションの出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {results_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    events_path = os.path.join(args.results_dir, "events.csv")
    out_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Hua et al. (2024) WarAgent 外交動態 可視化 ===")
    print(f"メトリクス: {metrics_path}")
    print(f"イベント:   {events_path}")
    print(f"出力先:     {out_dir}")
    print("-----------------------------------------")

    metrics = load_metrics(metrics_path)
    events = load_events(events_path)
    n_rounds = metrics["round"].nunique()
    print(f"      {n_rounds} ラウンド分の指標, {len(events)} 行動イベント")
    print("[1/1] 外交動態図 (同盟網・Board 遷移・指標・紛争規模) を保存中 ...")
    save_war_dynamics(metrics, events, os.path.join(out_dir, "war_dynamics.png"))

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
