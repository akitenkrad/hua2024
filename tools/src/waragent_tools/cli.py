"""waragent-tools — Hua et al. (2024) WarAgent シミュレーション ツール統合 CLI．

Usage:
    waragent-tools visualize [...]
    waragent-tools visualize-sweep [...]
    waragent-tools show-experiment-settings [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

`reproduce` (論文 Table 2-5 の一括再現・反実仮想分析) は Phase 3 で実装予定 (未提供)．
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="waragent-tools",
        description="Hua et al. (2024) WarAgent 世界大戦外交シミュレーション 可視化・分析ツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "visualize",
        help="単一実行結果 (同盟ネットワーク図・Board 遷移・指標時系列) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果 (開戦率ヒートマップ・同盟類似度 vs トリガー/スタンス) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from waragent_tools.visualize import main as run_main

        run_main(rest)
    elif command == "visualize-sweep":
        from waragent_tools.visualize_sweep import main as run_main

        run_main(rest)
    elif command == "show-experiment-settings":
        from waragent_tools.show_experiment_settings import main as run_main

        run_main(rest)
    else:
        # 未知のコマンドは argparse のエラーメッセージに委ねる
        parser.parse_args(argv)


if __name__ == "__main__":
    main()
