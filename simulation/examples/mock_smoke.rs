//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (metrics.csv / events.csv / run_metadata.json / config.json) と
//! Python 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で
//! 決定論的に国の行動を駆動し，本番 `run` と同じ writer で結果を書き出す．
//!
//! 4 カ国 (wwi-small) × 2 ラウンド × secretary_passes=1 の小シナリオで，A が B に
//! 公開宣戦布告 → B の同盟国がエスカレーション参戦するシナリオを擬似的に再現する．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;
use std::fs;

use chrono::Local;

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
use waragent_simulation::config::{Config, Scenario, Trigger};
use waragent_simulation::llm::wrap_client;
use waragent_simulation::simulation::{
    ensure_output_dir, run_with_client, save_events, save_metrics, save_run_metadata,
};

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        scenario: Scenario::WwiSmall,
        trigger: Trigger::ArchdukeAssassination,
        secretary_passes: 1,
        rounds: 2,
        // しきい値 2: A→C の 1 宣戦対だけでは «世界大戦勃発» とせず両ラウンドを走らせ，
        // 同盟形成・総動員・宣戦が共存する状態を観測する (war_outbreak は false のまま)．
        war_threshold: 2,
        seed: Some(42),
        output_dir: output_dir.clone(),
        ..Config::default()
    };

    // 行動を国ごとに固定する mock．wwi-small は A,B (同盟),C,D．
    // - Country A: B (id 1) ではなく C (id 2) へ公開宣戦布告 → C の動きを誘発．
    // - Country C: D (id 3) へ公開同盟要請．
    // - Country D: C (id 2) へ公開同盟要請 (相互 → 同盟成立)．
    // - Country B: 総動員 (mobilize)．
    // 国名はプロンプトに "Your country: Country X" (decision) と
    // "## Country: Country X" (secretary 検証) の両方で現れるため，両方を見て同じ
    // 行動を返す (secretary が decision を上書きしないように)．
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        let is = |letter: &str| {
            prompt.contains(&format!("Your country: Country {letter}"))
                || prompt.contains(&format!("## Country: Country {letter}"))
        };
        if is("A") {
            "{\"action\": \"declare_war\", \"target\": 2, \"publicity\": \"public\"}".to_string()
        } else if is("C") {
            "{\"action\": \"alliance\", \"target\": 3, \"publicity\": \"public\"}".to_string()
        } else if is("D") {
            "{\"action\": \"alliance\", \"target\": 2, \"publicity\": \"public\"}".to_string()
        } else if is("B") {
            "{\"action\": \"mobilize\"}".to_string()
        } else {
            "{\"action\": \"wait\"}".to_string()
        }
    });
    let client = wrap_client(backend, PromptCache::in_memory());

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_events(&result.event_log, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    println!("mock smoke wrote: {output_dir}");
    let last = result.metrics_history.last().unwrap();
    println!(
        "final round={} alliance_mi={:.3} declaration_jaccard={:.3} mobilization_jaccard={:.3} n_conflicts={} war_outbreak={} cold_war={}",
        result.final_round,
        last.alliance_mi,
        last.declaration_jaccard,
        last.mobilization_jaccard,
        last.n_conflicts,
        result.war_outbreak,
        result.cold_war_flag,
    );
    println!("events: {} rows", result.event_log.len());
    for e in &result.event_log {
        println!(
            "  round {} actor {} {} {:?} [{}]",
            e.round, e.actor, e.action, e.target, e.publicity
        );
    }
}
