"""waragent-tools — Hua et al. (2024) WarAgent シミュレーション ツール統合 CLI．

Usage:
    waragent-tools visualize [...]
    waragent-tools visualize-sweep [...]
    waragent-tools show-experiment-settings [...]
    waragent-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

`reproduce` は論文 (Hua et al. 2024) の Table 2-5 ヘッドライン指標を一括再現する
(トリガー強度別の開戦頻度・エスカレーション・同盟分極化; `--run --mock --quick` で
オフライン検証可能)．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示の
実体 (visualize / visualize_sweep / show_experiment_settings) は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="waragent-tools",
    description="Hua et al. (2024) WarAgent 世界大戦外交シミュレーション 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (同盟ネットワーク図・Board 遷移・指標時系列) の可視化",
            "waragent_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (開戦率ヒートマップ・同盟類似度 vs トリガー/スタンス) の可視化",
            "waragent_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
            "waragent_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "論文 Table 2-5 ヘッドライン指標の一括再現 (トリガー別 開戦/冷戦/同盟分極化 + figure)",
            "waragent_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
