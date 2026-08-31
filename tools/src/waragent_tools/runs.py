"""run ディレクトリの解決と読み出し (waragent 固有の共通部分)．

runvault の run ディレクトリは `metrics.csv` を long 形式 (`run_uid,step,
step_unit,scope,name,value`) で，国の行動ログを `events.jsonl` の名前空間つき
イベント `x.hua2024.action` で持つ．可視化スクリプトはどれもこの 2 つを
«ラウンド × 指標» の表と «1 行 1 行動» の表に直してから使うので，直し方を
ここ 1 箇所に集める．

runvault 以前の legacy な出力 (`results/<timestamp>/` の wide な `metrics.csv` と
`events.csv`) もそのまま読める — ディスクに残っている結果を読めなくする理由は
ないので，`--results-dir` に直接渡せば従来どおり扱える．
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from runvault.read import (
    config_parameters,
    events_table,
    load_run_meta,
    runvault_path,
    sweep_children,
)

# runvault 上の実験名 (Rust 側 `record::EXPERIMENT` と同じ値)．
EXPERIMENT = "waragent"
# 国の行動ログの種別 (Rust 側 `record::ACTION_EVENT` と同じ値)．
ACTION_EVENT = "x.hua2024.action"

EVENT_COLUMNS = ["round", "actor", "action", "target", "publicity"]


def resolve_run_dir(
    results_dir: str | None,
    results_root: str = "results",
    subcommand: str = "run",
    standalone: bool = True,
) -> str:
    """どの run を見るか．未指定なら runvault が答える．

    `results/` を自分で走査して新しそうなディレクトリを当てにいくことはしない．
    掃引の子も `subcommand=run` なので，単独の run が欲しいときは `standalone`
    を付ける (付けないと «最後に走った子» が返る)．

    legacy の `results/latest` のようなシンボリックリンクは実体に解決する．
    """
    if results_dir is None:
        return runvault_path(
            EXPERIMENT,
            results_root=results_root,
            subcommand=subcommand,
            standalone=standalone,
        )
    p = Path(results_dir)
    return str(Path(os.path.realpath(p)) if p.is_symlink() else p)


def _read_metrics_csv(run_dir: str) -> pd.DataFrame:
    """`metrics.csv` を «書かれたとおりの» f64 で読む．

    pandas の既定パーサは f64 を 1 ULP 落とすことがある
    (`0.05376245048607731` → `0.0537624504860773`)．記録された値と読み出した値が
    最後の桁で食い違うと，移行前後の突き合わせも過去の run との比較も成り立たなく
    なるので `float_precision="round_trip"` で読む．`runvault.read.metrics_wide`
    は既定パーサを使い，他リポジトリが依存しているので変更しない — こちらで読む．
    """
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    return pd.read_csv(path, float_precision="round_trip")


def _is_long(df: pd.DataFrame) -> bool:
    return {"name", "value", "step"}.issubset(df.columns)


def load_metrics(run_dir: str) -> pd.DataFrame:
    """ラウンドごとの指標を 1 ラウンド 1 行の表として読む．

    runvault の `metrics.csv` は long 形式なので横に倒す．時間軸の列名は runvault
    では `step` だが，本モデルの表記は `round` なのでこちら側の呼び名に揃えてから
    返す (legacy の wide な `metrics.csv` はもともと `round` 列を持つので
    そのまま返す)．
    """
    df = _read_metrics_csv(run_dir)
    if not _is_long(df):
        return df
    stepped = df[df["step"].notna()]
    return (
        stepped.pivot_table(index="step", columns="name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .astype({"step": int})
        .sort_values("step")
        .reset_index(drop=True)
        .rename(columns={"step": "round"})
    )


def run_scope(run_dir: str) -> dict[str, float]:
    """run 全体を 1 つの値で表す指標 (step を持たない行)．

    legacy の wide な `metrics.csv` にはこの行が無いので空の辞書を返す — そちらでは
    同じ値を `run_metadata.json` が持っている．
    """
    df = _read_metrics_csv(run_dir)
    if not _is_long(df) or df.empty:
        return {}
    rows = df[df["step"].isna()]
    return {str(r["name"]): float(r["value"]) for _, r in rows.iterrows()}


def load_events(run_dir: str) -> pd.DataFrame:
    """国の行動ログを 1 行動 1 行の表として読む．

    runvault の run では `events.jsonl` の `x.hua2024.action`，legacy では
    `events.csv`．どちらも `round, actor, action, target, publicity` の 5 列に
    揃えて返す．行動ログが 1 行も無い run もありうるので (`--rounds 0` 等)，
    無い場合は空表を返す．
    """
    legacy = os.path.join(run_dir, "events.csv")
    if os.path.exists(legacy):
        return pd.read_csv(legacy)
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        df = events_table(run_dir, kind=ACTION_EVENT)
    except SystemExit:
        # events.jsonl はあるが行動ログが 1 行も無い．
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return df.rename(columns={"t": "round"})[EVENT_COLUMNS]


def sweep_table(sweep_dir: str) -> pd.DataFrame:
    """1 行 1 実行の掃引サマリ表．

    runvault はこの表をディスクに持たない (旧 `sweep_summary.csv` はもう書かない)．
    掃引親の子 run から組み直す — 条件は子の `config.json` の `parameters`，
    試行番号とシードは `run.json` の `rng`，最終ラウンドの値は子の `metrics.csv`
    の最後のステップ，run 全体の値は step を持たない行が持つ．

    列は旧 `sweep_summary.csv` と同じで，`escalation_round` は勃発しなかった run
    では欠測 (`NaN`) になる — 記録側も «無い» ものは行を書かないので，0 で埋めない．

    legacy な掃引ディレクトリには `sweep_summary.csv` が残っているので，あれば
    そちらを読む (`null` を文字列のまま読む必要があるので trigger 系の列だけ
    NA 変換を止める)．
    """
    legacy = os.path.join(sweep_dir, "sweep_summary.csv")
    if os.path.exists(legacy):
        df = pd.read_csv(legacy)
        raw = pd.read_csv(legacy, keep_default_na=False, dtype=str)
        for col in ("trigger", "stance", "scenario"):
            if col in raw.columns:
                df[col] = raw[col]
        return df

    rows: list[dict] = []
    for child in sweep_children(sweep_dir):
        params = config_parameters(child) or {}
        meta = load_run_meta(child) or {}
        rng = meta.get("rng") or {}
        scope = run_scope(child)
        steps = load_metrics(child)
        last = steps.iloc[-1] if not steps.empty else None

        def at(name: str):
            return None if last is None or name not in steps.columns else float(last[name])

        rows.append(
            {
                "scenario": params.get("scenario"),
                "trigger": params.get("trigger"),
                "stance": params.get("stance_override"),
                "run": rng.get("replicate_index"),
                "seed": rng.get("master_seed"),
                "final_round": scope.get("final_round"),
                "war_outbreak": at("war_outbreak"),
                "escalation_round": scope.get("escalation_round"),
                "n_conflicts": at("n_conflicts"),
                "cold_war_flag": scope.get("cold_war_flag"),
                "final_alliance_mi": at("alliance_mi"),
                "final_declaration_jaccard": at("declaration_jaccard"),
                "final_mobilization_jaccard": at("mobilization_jaccard"),
                "cache_hit_rate": scope.get("llm_cache_hit_rate"),
                "run_dir": child,
            }
        )
    if not rows:
        raise SystemExit(
            f"エラー: この掃引親に子 run がありません: {sweep_dir}\n"
            "  子は lineage.parent_run_uid で親を指す．親子が同じ results root に"
            "いるか確認してください．"
        )
    return pd.DataFrame(rows).sort_values(["trigger", "stance", "run"]).reset_index(drop=True)
