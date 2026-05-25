//! Hua et al. (2024) "War and Peace (WarAgent)" — 再現実験の CLI エントリポイント．
//!
//! `run`   : 単一設定で LLM 駆動の国エージェント外交 ABM を実行する．
//! `sweep` : トリガー強度 × スタンス を走査し，開戦率・同盟 MI 等を
//!           `sweep_summary.csv` に集計する．
//!
//! Phase 3 の `reproduce` (論文 Table 2-5 一括再現・反実仮想分析・WWII/戦国時代
//! シナリオ・脱匿名化) は未実装 (拡張点)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use waragent_simulation::config::{
    derive_run_seed, parse_scenario, parse_stance, parse_trigger, Config, LlmSettings, Scenario,
    Trigger,
};
use waragent_simulation::simulation::{
    ensure_output_dir, run, save_events, save_metrics, save_run_metadata, SimulationResult,
};
use waragent_simulation::world::Stance;

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "waragent",
    about = "Hua et al. (2024) WarAgent: LLM-based Multi-Agent Simulation of World Wars — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で LLM 駆動の国エージェント外交 ABM を実行する．
    Run(RunArgs),
    /// トリガー強度 × スタンス を走査し，開戦率・同盟 MI を集計する．
    Sweep(SweepArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// シナリオ (wwi / wwi-small)．
    #[arg(long, default_value = "wwi")]
    scenario: String,

    /// トリガー (null / naval-incident / dardanelles / archduke-assassination)．
    #[arg(long, default_value = "archduke-assassination")]
    trigger: String,

    /// 全国に上書きするスタンス (conservative / neutral / aggressive; 省略でシナリオ既定)．
    #[arg(long)]
    stance: Option<String>,

    /// 秘書検証パス数 (各国の最終行動を検証する回数; LLM 呼び出しを有界化)．
    #[arg(long, default_value_t = 1)]
    secretary_passes: usize,

    /// 最大ラウンド数 (論文評価は round 6)．
    #[arg(long, default_value_t = 6)]
    rounds: usize,

    /// 世界大戦勃発の宣戦布告対しきい値．
    #[arg(long, default_value_t = 3)]
    war_threshold: usize,

    /// 独立試行数 (各試行は derive により独立化)．
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0)．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// シナリオ．
    #[arg(long, default_value = "wwi")]
    scenario: String,

    /// カンマ区切りのトリガー候補．
    #[arg(long, default_value = "null,naval-incident,dardanelles")]
    trigger_values: String,

    /// カンマ区切りのスタンス候補．
    #[arg(long, default_value = "conservative,aggressive")]
    stance_values: String,

    /// 秘書検証パス数．
    #[arg(long, default_value_t = 1)]
    secretary_passes: usize,

    /// 最大ラウンド数．
    #[arg(long, default_value_t = 6)]
    rounds: usize,

    /// 世界大戦勃発しきい値．
    #[arg(long, default_value_t = 3)]
    war_threshold: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// 乱数シード基点．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// `sweep_summary.csv` の 1 行．
#[derive(serde::Serialize)]
struct SweepRow {
    scenario: String,
    trigger: String,
    stance: String,
    run: usize,
    seed: u64,
    final_round: usize,
    war_outbreak: u8,
    escalation_round: Option<u64>,
    n_conflicts: u64,
    cold_war_flag: u8,
    final_alliance_mi: f64,
    final_declaration_jaccard: f64,
    final_mobilization_jaccard: f64,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    scenario: String,
    trigger_values: Vec<String>,
    stance_values: Vec<String>,
    secretary_passes: usize,
    rounds: usize,
    war_threshold: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let scenario = parse_scenario(&args.scenario).unwrap_or_else(|e| panic!("{e}"));
    let trigger = parse_trigger(&args.trigger).unwrap_or_else(|e| panic!("{e}"));
    let stance_override: Option<Stance> = args
        .stance
        .as_deref()
        .map(|s| parse_stance(s).unwrap_or_else(|e| panic!("{e}")));

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&output_dir);

    println!("=== Hua et al. (2024) WarAgent 世界大戦外交 再現実験 ===");
    println!(
        "シナリオ: {} | トリガー: {} | スタンス: {} | ラウンド: {} | 秘書pass: {} | 試行: {}",
        scenario.label(),
        trigger.label(),
        stance_override
            .map(|s| s.label())
            .unwrap_or("scenario-default"),
        args.rounds,
        args.secretary_passes,
        args.runs,
    );
    println!(
        "LLM: temp={} llm_seed={} cache={} | seed: {:?}",
        args.llm_temperature, args.llm_seed, args.cache_path, args.seed
    );
    println!("出力先: {output_dir}");
    println!("-------------------------------------------------");

    let base_seed = args.seed.unwrap_or(42);
    let mut last_result: Option<SimulationResult> = None;
    let mut outbreak_count = 0usize;
    let mut cold_war_count = 0usize;

    for run_idx in 0..args.runs.max(1) {
        let seed = derive_run_seed(base_seed, run_idx);
        let cfg = Config {
            scenario,
            trigger,
            stance_override,
            secretary_passes: args.secretary_passes,
            rounds: args.rounds,
            war_threshold: args.war_threshold,
            seed: Some(seed),
            llm: LlmSettings {
                temperature: args.llm_temperature,
                seed: args.llm_seed,
                cache_path: Some(args.cache_path.clone()),
            },
            output_dir: output_dir.clone(),
        };

        let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {e}"));
        if result.war_outbreak {
            outbreak_count += 1;
        }
        if result.cold_war_flag {
            cold_war_count += 1;
        }

        // 最後の試行の詳細を保存する (代表 run)．
        if run_idx + 1 == args.runs.max(1) {
            save_metrics(&result.metrics_history, &output_dir);
            save_events(&result.event_log, &output_dir);
            save_run_metadata(&result, &cfg, &output_dir);
            let path = format!("{output_dir}/config.json");
            write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
            last_result = Some(result);
        }
    }

    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    let runs = args.runs.max(1);
    println!(
        "開戦発生: {}/{} ({:.1}%) | 冷戦 (緊張のみ): {}/{}",
        outbreak_count,
        runs,
        100.0 * outbreak_count as f64 / runs as f64,
        cold_war_count,
        runs,
    );
    if let Some(result) = &last_result {
        if let Some(last) = result.metrics_history.last() {
            println!(
                "最終 同盟MI: {:.3} | 宣戦Jaccard: {:.3} | 総動員Jaccard: {:.3} | 紛争数: {} | 総動員: {}",
                last.alliance_mi,
                last.declaration_jaccard,
                last.mobilization_jaccard,
                last.n_conflicts,
                last.n_mobilized,
            );
        }
        println!(
            "勃発ラウンド: {:?} | LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
            result.escalation_round,
            result.metadata.total(),
            result.metadata.cache_hits(),
            result.metadata.cache_hit_rate() * 100.0,
            result.llm_model,
        );
    }
    println!("メトリクス → {output_dir}/metrics.csv");
    println!("イベント   → {output_dir}/events.csv");
    println!("LLM メタ   → {output_dir}/run_metadata.json");
    println!("設定       → {output_dir}/config.json");
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let scenario: Scenario = parse_scenario(&args.scenario).unwrap_or_else(|e| panic!("{e}"));
    let triggers: Vec<Trigger> = split_csv(&args.trigger_values)
        .iter()
        .map(|s| parse_trigger(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let stances: Vec<Stance> = split_csv(&args.stance_values)
        .iter()
        .map(|s| parse_stance(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = triggers.len() * stances.len() * args.runs;

    println!("=== Hua et al. (2024) WarAgent 感度分析 (トリガー × スタンス) ===");
    println!(
        "シナリオ: {} | トリガー {} 種 × スタンス {} 種 | 試行: {} | 合計: {} 実行",
        scenario.label(),
        triggers.len(),
        stances.len(),
        args.runs,
        n_total,
    );
    println!("出力先: {sweep_dir}");
    println!("-----------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &trigger in &triggers {
        for &stance in &stances {
            for run_idx in 0..args.runs {
                let seed = sweep_seed(args.seed, &trigger, &stance, run_idx);
                let cfg = Config {
                    scenario,
                    trigger,
                    stance_override: Some(stance),
                    secretary_passes: args.secretary_passes,
                    rounds: args.rounds,
                    war_threshold: args.war_threshold,
                    seed: Some(seed),
                    llm: LlmSettings {
                        temperature: args.llm_temperature,
                        seed: args.llm_seed,
                        cache_path: Some(args.cache_path.clone()),
                    },
                    output_dir: sweep_dir.clone(),
                };

                let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {e}"));
                summary_rows.push(summarize(&result, scenario, trigger, stance, run_idx, seed));
                done += 1;
            }
            println!(
                "[{}/{}] trigger={} stance={} 完了 ({} 試行)",
                done,
                n_total,
                trigger.label(),
                stance.label(),
                args.runs,
            );
        }
    }

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{sweep_dir}/sweep_summary.csv");
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            scenario: scenario.label().to_string(),
            trigger_values: triggers.iter().map(|t| t.label().to_string()).collect(),
            stance_values: stances.iter().map(|s| s.label().to_string()).collect(),
            secretary_passes: args.secretary_passes,
            rounds: args.rounds,
            war_threshold: args.war_threshold,
            runs: args.runs,
            seed: args.seed,
            llm_temperature: args.llm_temperature,
            llm_seed: args.llm_seed,
        };
        let path = format!("{sweep_dir}/sweep_config.json");
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{timestamp}_sweep"));

    println!("===========================================================");
    println!("スイープ完了: {n_total} 実行");
    println!("-----------------------------------------------------------");
    println!("トリガー別の開戦発生頻度 / 平均 同盟MI:");
    for &trigger in &triggers {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| r.trigger == trigger.label())
            .collect();
        if rows.is_empty() {
            continue;
        }
        let outbreak_freq =
            rows.iter().filter(|r| r.war_outbreak == 1).count() as f64 / rows.len() as f64;
        let avg_mi = rows.iter().map(|r| r.final_alliance_mi).sum::<f64>() / rows.len() as f64;
        println!(
            "  trigger={} → 開戦 = {:.1}% | 同盟MI = {:.3}",
            trigger.label(),
            outbreak_freq * 100.0,
            avg_mi
        );
    }
    println!("-----------------------------------------------------------");
    println!("サマリ → {sweep_dir}/sweep_summary.csv");
    println!("設定   → {sweep_dir}/sweep_config.json");
}

/// sweep の試行シードを派生する (トリガー・スタンス・試行 index で独立化)．
fn sweep_seed(base: u64, trigger: &Trigger, stance: &Stance, run_idx: usize) -> u64 {
    // ラベル文字列を簡易ハッシュして派生引数にする (決定論)．
    let th = label_hash(trigger.label());
    let sh = label_hash(stance.label());
    socsim_core::derive_seed(base, &[th, sh, run_idx as u64])
}

/// 文字列の決定論的ハッシュ (FNV-1a 風)．
fn label_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 1 実行結果を sweep の 1 行に集約する．
fn summarize(
    result: &SimulationResult,
    scenario: Scenario,
    trigger: Trigger,
    stance: Stance,
    run_idx: usize,
    seed: u64,
) -> SweepRow {
    let last = result.metrics_history.last();
    SweepRow {
        scenario: scenario.label().to_string(),
        trigger: trigger.label().to_string(),
        stance: stance.label().to_string(),
        run: run_idx,
        seed,
        final_round: result.final_round,
        war_outbreak: if result.war_outbreak { 1 } else { 0 },
        escalation_round: result.escalation_round,
        n_conflicts: result.n_conflicts,
        cold_war_flag: if result.cold_war_flag { 1 } else { 0 },
        final_alliance_mi: last.map(|m| m.alliance_mi).unwrap_or(0.0),
        final_declaration_jaccard: last.map(|m| m.declaration_jaccard).unwrap_or(0.0),
        final_mobilization_jaccard: last.map(|m| m.mobilization_jaccard).unwrap_or(0.0),
        cache_hit_rate: result.metadata.cache_hit_rate(),
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
    }
}
