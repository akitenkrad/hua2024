//! Hua et al. (2024) "War and Peace (WarAgent): Large Language Model-based
//! Multi-Agent Simulation of World Wars" (arXiv:2311.17227) の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した LLM 駆動の «国エージェント外交ラウンド ABM»
//! (WWI 8 カ国の同盟・宣戦布告・総動員動学) の公開 API を提供する．設定/シナリオ
//! (`config`)・世界状態 (`world`)・Board/Stick と Translate (`board`)・LLM
//! クライアント層 (`llm`)・プロンプト生成と応答パース (`prompts`)・更新メカニズム
//! (`mechanisms`)・実行ドライバ (`simulation`)・集計メトリクス (`metrics`) を
//! モジュールとして公開し，バイナリ (`waragent`) と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (シナリオ/Board 初期化・活性化順・publicity 伝播・同盟/宣戦の
//! 解決・エスカレーション・Board 更新・全指標) は seed から bit 単位で決定論的で
//! ある．LLM レイヤ (4 ステップ誘導推論 + 秘書検証) は socsim の bit 再現性の
//! **外側** にあり，`socsim-llm` のキャッシュ + `temperature=0` + `seed` 固定で
//! 擬似決定論化する．設計書 §4.2/§7 は当初 `reqwest` + `sha2` を挙げていたが，
//! 本スイートは li2024 / zhao2024 / ren2024 / gao2023 と統一して `socsim-llm`
//! (issue #21/#26) に標準化したため `reqwest` / `sha2` は使わない (socsim-llm が
//! HTTP とプロンプトハッシュを所有する)．
//!
//! # 関係構造の表現
//!
//! 国は少数 (WWI=8) で空間格子を持たない．自国視点の Board 関係 (相手国 →
//! W/M/T/P) を `BTreeMap<(AgentId, AgentId), Relation>` 相当の明示的行列
//! (`board::Board`) で保持し，これを **source of truth** とする (部分情報の原則)．
//! `socsim-net` は同盟クラスタ抽出/可視化の «導出ビュー» 用に依存に含めるが，
//! 指標は Board から直接 union-find で計算するため必須ではない．

pub mod board;
pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;
