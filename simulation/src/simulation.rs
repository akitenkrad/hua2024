//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で世界初期化
//!   (シナリオ国・初期 Board) の init RNG を，`derive_seed(root, &[1])` で engine
//!   RNG (= 活性化順) を派生する．bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き
//!   Ollama→OpenAI フォールバッククライアントに閉じ込め，`temperature=0`/`seed`
//!   固定 + プロンプト→応答キャッシュで擬似決定論化する．モデル・endpoint・
//!   温度・seed・cache-hit を `run_metadata.json` に記録する．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::rc::Rc;

use csv::Writer;
use serde::Serialize;

use socsim_core::{derive_seed, SimClock, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};

use crate::config::{build_countries, Config};
use crate::llm::{build_live_client, WarClient};
use crate::mechanisms::{
    BoardUpdateMechanism, CountryDecisionMechanism, DiplomacyMechanism, EscalationMechanism,
    SharedClient, SharedMetadata, SharedMetrics, SituationMechanism,
};
use crate::metrics::RoundMetric;
use crate::world::{Event, WarWorld};

/// 世界初期化用 RNG ラベル (シナリオ国・初期 Board)．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン (= 活性化順) 用 RNG ラベル．
const RNG_ENGINE: u64 = 1;

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// ラウンド指標の履歴 (metrics.csv の行)．
    pub metrics_history: Vec<RoundMetric>,
    /// 行動イベントログ (events.csv)．
    pub event_log: Vec<Event>,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名．
    pub llm_model: String,
    /// LLM endpoint (primary)．
    pub llm_endpoint: String,
    /// 実行したラウンド数 (= 完了ステップ数)．
    pub final_round: usize,
    /// 世界大戦が勃発したか．
    pub war_outbreak: bool,
    /// 初回勃発に至ったラウンド (なければ None)．
    pub escalation_round: Option<u64>,
    /// 終了時点の宣戦布告 (W) 対の総数．
    pub n_conflicts: u64,
    /// 冷戦フラグ (開戦せず緊張のみ: 動員 or 同盟変化はあるが war_outbreak=false)．
    pub cold_war_flag: bool,
}

/// 世界状態を初期化する (シナリオ国 + 初期 Board)．
///
/// 国とプロフィールはシナリオ定数 (匿名化済み) から決定論的に作る．init RNG は
/// 将来の «プロフィール摂動» 等の拡張点として受け取るが，Phase 1 では固定データの
/// ため消費しない (決定論の seed 規約は維持する)．
pub fn init_world(cfg: &Config, _rng: &mut SimRng) -> WarWorld {
    let (countries, boards) = build_countries(cfg.scenario, cfg.stance_override);
    let mut inbox = BTreeMap::new();
    for id in countries.keys() {
        inbox.insert(*id, Vec::new());
    }
    WarWorld {
        clock: SimClock::new(cfg.rounds as u64),
        countries,
        boards,
        pending_actions: Vec::new(),
        inbox,
        event_log: Vec::new(),
        round: 0,
        trigger: cfg.trigger.description().map(|s| s.to_string()),
        trigger_injected: false,
    }
}

/// シミュレーションを実行する (本番 LLM クライアントを構築して駆動)．
pub fn run(cfg: &Config) -> std::result::Result<SimulationResult, String> {
    let client =
        build_live_client(&cfg.llm).map_err(|e| format!("LLM クライアント構築に失敗: {e}"))?;
    run_with_client(cfg, client)
}

/// 与えられた [`WarClient`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．
pub fn run_with_client(
    cfg: &Config,
    client: WarClient,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_metrics: SharedMetrics = Rc::new(RefCell::new(Vec::new()));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(SituationMechanism))
        .add_mechanism(Box::new(CountryDecisionMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            cfg.secretary_passes,
        )))
        .add_mechanism(Box::new(DiplomacyMechanism))
        .add_mechanism(Box::new(EscalationMechanism))
        .add_mechanism(Box::new(BoardUpdateMechanism::new(
            cfg.rounds as u64,
            cfg.war_threshold,
            Rc::clone(&shared_metrics),
        )))
        .build();

    // 勃発ラウンドを観測する．
    let mut escalation_round: Option<u64> = None;
    let mut final_round = 0usize;
    sim.run_observed(|report| {
        final_round = report.t as usize;
        if escalation_round.is_none() {
            if let Some(outbreak) = report.scratch.get::<bool>("war_outbreak") {
                if *outbreak {
                    // report.t は 1 始まり; ラウンドは 0 始まり．
                    escalation_round = Some(report.t.saturating_sub(1));
                }
            }
        }
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    // キャッシュ保存 (cache_path 指定時)．
    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let metrics_history = shared_metrics.borrow().clone();
    let metadata = shared_meta.borrow().clone();

    // 最終的な world は sim が所有しているため，集計は metrics_history の最後の行から取る．
    let last = metrics_history.last();
    let war_outbreak = last.map(|m| m.war_outbreak == 1).unwrap_or(false);
    let n_conflicts = last.map(|m| m.n_conflicts).unwrap_or(0);
    let any_mobilized_or_alliance_change = metrics_history.iter().any(|m| {
        m.n_mobilized > 0
            || m.n_conflicts > 0
            || m.n_alliance_clusters as usize != initial_clusters(cfg)
    });
    // 冷戦: 開戦には至らないが緊張 (動員 or 同盟/関係の動き) はある．
    let cold_war_flag = !war_outbreak && any_mobilized_or_alliance_change;

    // event_log を sim の world 参照から複製する (run_observed 後)．
    let event_log = sim.world().event_log.clone();

    Ok(SimulationResult {
        metrics_history,
        event_log,
        metadata,
        llm_model,
        llm_endpoint,
        final_round,
        war_outbreak,
        escalation_round,
        n_conflicts,
        cold_war_flag,
    })
}

/// シナリオ初期の同盟クラスタ数 (冷戦判定の基準)．
fn initial_clusters(cfg: &Config) -> usize {
    let (countries, boards) = build_countries(cfg.scenario, cfg.stance_override);
    let tmp = WarWorld {
        clock: SimClock::new(1),
        countries,
        boards,
        pending_actions: Vec::new(),
        inbox: BTreeMap::new(),
        event_log: Vec::new(),
        round: 0,
        trigger: None,
        trigger_injected: false,
    };
    crate::metrics::alliance_partition(&tmp).len()
}

/// ラウンド指標を CSV に保存する．
pub fn save_metrics(metrics: &[RoundMetric], output_dir: &str) {
    let path = format!("{}/metrics.csv", output_dir);
    let file = File::create(&path).expect("metrics.csv の作成に失敗");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    for m in metrics {
        wtr.serialize(m).expect("メトリクス書き込みに失敗");
    }
    wtr.flush().expect("フラッシュに失敗");
}

/// 行動イベントログを CSV に保存する．
pub fn save_events(events: &[Event], output_dir: &str) {
    let path = format!("{}/events.csv", output_dir);
    let file = File::create(&path).expect("events.csv の作成に失敗");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    for e in events {
        wtr.serialize(e).expect("イベント書き込みに失敗");
    }
    wtr.flush().expect("フラッシュに失敗");
}

/// `run_metadata.json` の構造体 (LLM モデル・endpoint・温度・seed・cache 統計)．
#[derive(Serialize)]
pub struct RunMetadataJson {
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub war_outbreak: bool,
    pub escalation_round: Option<u64>,
    pub n_conflicts: u64,
    pub cold_war_flag: bool,
    pub determinism_note: &'static str,
}

/// `run_metadata.json` を保存する．
pub fn save_run_metadata(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = RunMetadataJson {
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        war_outbreak: result.war_outbreak,
        escalation_round: result.escalation_round,
        n_conflicts: result.n_conflicts,
        cold_war_flag: result.cold_war_flag,
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (with temperature=0 and fixed seed) is the reproducibility \
                           mechanism. The socsim core (scenario/board init, activation order, \
                           publicity propagation, alliance/war resolution, escalation, board \
                           updates and all metrics) is deterministic given the seed. LLM calls \
                           per round = n_countries * (1 + secretary_passes).",
    };
    let path = format!("{}/run_metadata.json", output_dir);
    let file = File::create(&path).expect("run_metadata.json の作成に失敗");
    serde_json::to_writer_pretty(BufWriter::new(file), &meta)
        .expect("run_metadata.json の書き込みに失敗");
}

/// 出力ディレクトリを作成する．
pub fn ensure_output_dir(output_dir: &str) {
    fs::create_dir_all(output_dir).expect("出力ディレクトリの作成に失敗");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Scenario, Trigger};
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    /// A→declare war on B (id 1) public, everyone else waits.
    fn aggressor_client() -> WarClient {
        let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
            // Country A appears as "Your country: Country A" (decision) and
            // "## Country: Country A" (secretary verification); match both.
            if prompt.contains("Your country: Country A")
                || prompt.contains("## Country: Country A")
            {
                "{\"action\": \"declare_war\", \"target\": 1, \"publicity\": \"public\"}"
                    .to_string()
            } else {
                "{\"action\": \"wait\"}".to_string()
            }
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn small_cfg() -> Config {
        Config {
            scenario: Scenario::WwiSmall,
            trigger: Trigger::ArchdukeAssassination,
            rounds: 2,
            secretary_passes: 1,
            war_threshold: 1,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn run_produces_metrics_and_events() {
        let cfg = small_cfg();
        let r = run_with_client(&cfg, aggressor_client()).unwrap();
        assert!(!r.metrics_history.is_empty());
        assert!(r.final_round >= 1);
        // A declared war on B → at least one event of declare_war.
        assert!(r.event_log.iter().any(|e| e.action == "declare_war"));
    }

    #[test]
    fn deterministic_given_mock() {
        let cfg = small_cfg();
        let a = run_with_client(&cfg, aggressor_client()).unwrap();
        let b = run_with_client(&cfg, aggressor_client()).unwrap();
        let ma: Vec<u64> = a.metrics_history.iter().map(|m| m.n_conflicts).collect();
        let mb: Vec<u64> = b.metrics_history.iter().map(|m| m.n_conflicts).collect();
        assert_eq!(ma, mb, "同一シードは完全再現すべき");
    }
}
