[English](README.md) | **日本語**

# War and Peace (WarAgent): LLM ベース・マルチエージェント世界大戦シミュレーション — Hua et al. (2024)

Hua et al. (2024)「War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars」(arXiv:2311.17227) の，LLM 駆動の **国エージェント外交モデル** の再現実装である．少数の国エージェント (WWI = 8) が，国際危機の中で同盟・宣戦布告・条約・総動員をラウンドごとに推論する．各国は 6 次元プロフィール (Leadership / Military / Resources / Historical Background / Key Policy / Public Morale) を持ち，7 行動空間 (Wait / 総動員 / 宣戦布告 / 軍事同盟 / 不干渉条約 / 和平 / メッセージ) から 1 ラウンドに 1 行動を選ぶ．膨大なマルチエージェント文脈は **Board** (対外関係 W/M/T/P) と **Stick** (対内状態: 総動員) を短いパラグラフに Translate して圧縮し，国名は **匿名化** (Country A..H) して LLM が史実を «記憶» で再生するのを防ぐ．決定論的な [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) コアが Board/関係の解決・エスカレーション・全指標を担い，非決定的な LLM レイヤは単一の Decision メカニズムに閉じ込め，`socsim-llm` クレット (プロンプト→応答キャッシュ + `temperature=0` + 固定 seed) で擬似決定論化する．

## 二層決定論 (最初に読むこと)

LLM 出力は socsim の bit 再現性の **外側** にある．本設計は二層に分ける:

- **決定論的 socsim コア** — シナリオ/Board 初期化・活性化順・publicity に基づく伝播・同盟/宣戦/条約の解決・エスカレーション (同盟国の参戦)・Board と総動員の更新・全指標 (同盟相互情報量・宣戦/総動員 Jaccard・開戦判定・勃発ラウンド)．seed から bit 単位で再現する (`ctx.rng`, ChaCha20 `SimRng`)．
- **非決定的 LLM レイヤ** — 単一の `Decision` メカニズム (`CountryDecisionMechanism`)．各国が 4 ステップ誘導推論 (同盟候補特定 → 敵対候補認識 → 推奨行動 → 最終行動) と秘書検証 (回数指定可) を実行する．`socsim-llm` の `CachingClient` (`hash(prompt+model)` → 応答キャッシュ)・`temperature=0`・固定 seed で擬似決定論化する．プロバイダ順は **Ollama 第一 → OpenAI フォールバック** (`FallbackClient`)．

再現性の本体はモデルではなく **キャッシュ** である．ウォームキャッシュは同一応答を再生するため，再実行はコスト 0 かつ安定する．**1 ラウンドあたりの LLM 呼び出し回数 = 国数 × (1 + secretary_passes)**．`--secretary-passes` は既定 `1` で呼び出し予算を有界化する．各 run は `run_metadata.json` にモデル・endpoint・温度・seed・cache-hit 率を記録する．ローカル既定モデル (`llama3.2`) は論文の `GPT-4`/`Claude-2` と異なるため，再現忠実度は **中程度 (△〜○)**: «傾向» (戦争はおおむね勃発する; null トリガーは冷戦に留まる; 同盟 MI はランダムより高い) を目標とし，Table 2 の絶対値の一致は狙わない．

> 本プロジェクトは LLM レイヤを `socsim-llm` クレットに統一し，`reqwest` / `sha2` は使わない (HTTP とプロンプトハッシュは socsim-llm が所有する)．設計書 §4.2/§7 を上書きし，li2024 / zhao2024 / ren2024 / gao2023 と統一する．

## 関係構造

国は少数なので，各国固有の **Board** 関係 (`(owner, other) → W/M/T/P`) を明示的な `BTreeMap` 行列で保持し，これを **source of truth** とする (部分情報の原則)．加えて，クラスタ抽出/可視化のためにグローバルな無向同盟グラフを `socsim-net::SocialNetwork` として導出する．Board 上の union-find 分割と，このネットワークの連結成分数は一致する (相互検算)．空間格子は無いため `socsim-grid` は依存に含めない．

## インストールとクイックスタート

```bash
# Rust シミュレーションをビルド (socsim-net + socsim-llm を含む socsim を取得)
cargo build --release

# ローカル Ollama を起動しモデルを取得しておく．例:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# OpenAI フォールバック (任意):
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini

# 小さなスモーク (4 カ国・2 ラウンド) — 実 LLM 経路の確認用に安価:
cargo run --release -- run --scenario wwi-small --rounds 2 --runs 1 --secretary-passes 1 --seed 42

# 論文規模の WWI 基本実験 (8 カ国):
#   cargo run --release -- run --scenario wwi --trigger archduke-assassination --rounds 6 --runs 7 --seed 42

# Python 可視化ツールをインストール (workspace ルートで)
uv sync

# 直近実行の可視化 (同盟ネットワーク・Board 遷移・指標時系列)
uv run waragent-tools visualize

# 設定値と LLM メタデータの確認
uv run waragent-tools show-experiment-settings --results-dir results/latest
```

### オフライン (LLM 不要) スモーク

ラウンドループ・出力 writer・Python 可視化は，スクリプト化した mock クライアントでライブ LLM 無しに検証できる:

```bash
cargo run --release --example mock_smoke -- results
uv run waragent-tools visualize
```

### 感度分析 (トリガー × スタンス)

```bash
cargo run --release -- sweep \
    --scenario wwi \
    --trigger-values null,naval-incident,dardanelles \
    --stance-values conservative,aggressive \
    --rounds 6 --runs 7 --seed 42

uv run waragent-tools visualize-sweep
```

## 出力

各 `run` は `results/{timestamp}/` を書き出す (`results/latest` シンボリックリンク付き):

| ファイル | 内容 |
|---|---|
| `config.json` | 実行設定 |
| `metrics.csv` | ラウンドごとの指標 (`alliance_mi`, `declaration_jaccard`, `mobilization_jaccard`, `n_conflicts`, `n_mobilized`, `n_alliance_clusters`, `war_outbreak`) |
| `events.csv` | 行動ログ (`round, actor, action, target, publicity`) |
| `run_metadata.json` | LLM モデル / endpoint / 温度 / seed / cache-hit 率 + マクロ帰結 |

`sweep` は `results/{timestamp}_sweep/` に `sweep_config.json` と `sweep_summary.csv` を書き出す．

## ドキュメント

- [アーキテクチャ](docs/architecture.ja.md) ([English](docs/architecture.md))
- [CLI リファレンス](docs/cli.ja.md) ([English](docs/cli.md))
- [可視化](docs/visualization.ja.md) ([English](docs/visualization.md))

## スコープ

- **Phase 1** (コアモデル: 国エージェント + Board + Board/Stick + イベントログ; 6 フェーズ上の 5 メカニズム; LLM 決定レイヤ + キャッシュ; WWI 基本 run) — 完了．
- **Phase 2** (`sweep` トリガー × スタンス; `visualize` / `visualize-sweep` / `show-experiment-settings`) — 完了．
- **Phase 3** (`reproduce` による Table 2–5 一括再現 + 反実仮想分析・WWII / 戦国時代シナリオ・脱匿名化) — 保留．拡張点を残してある: `Scenario` 列挙・設定可能な `secretary_passes`・`config.rs` の脱匿名化対応表 (A=Germany, B=Austria-Hungary, …)．

## 参考文献

Hua, W., Fan, L., Li, L., Mei, K., Ji, J., Ge, Y., Hemphill, L., & Zhang, Y. (2023/2024). War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars. *arXiv preprint* arXiv:2311.17227.

## ライセンス

MIT — [LICENSE](LICENSE) を参照．

---
*This file was generated by Claude Code.*
