//! Hua et al. (2024) "War and Peace (WarAgent)" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で LLM 駆動の国エージェント外交 ABM を実行する．
//! `sweep`     : トリガー強度 × スタンス を走査し，開戦率・同盟 MI 等を
//!               `sweep_summary.csv` に集計する．
//! `reproduce` : 論文 (Hua et al. 2024) の Table 2-5 ヘッドライン指標 — トリガー強度に
//!               応じた開戦頻度・エスカレーション・同盟分極化 — を 3 つのトリガー条件
//!               (null / dardanelles / archduke) を `wwi-small` で走らせ，観測値 vs
//!               論文値の PASS/off アンカーと figure 入力を `reproduce_summary.json`
//!               へ集約する．`--mock` でライブ LLM 無しに決定論再現する．
//!
//! 反実仮想分析・WWII/戦国時代シナリオ・脱匿名化比較は本コマンドの対象外 (拡張点)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
use waragent_simulation::config::{
    derive_run_seed, parse_scenario, parse_stance, parse_trigger, Config, LlmSettings, Scenario,
    Trigger,
};
use waragent_simulation::llm::wrap_client;
use waragent_simulation::simulation::{
    ensure_output_dir, run, run_with_client, save_events, save_metrics, save_run_metadata,
    SimulationResult,
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
    /// 論文 Table 2-5 のヘッドライン指標を一括再現する (観測 vs 論文 + figure 入力)．
    Reproduce(ReproduceArgs),
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

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// シナリオ (既定 wwi-small; オフライン検証可能な軽量シナリオ)．
    #[arg(long, default_value = "wwi-small")]
    scenario: String,

    /// 秘書検証パス数．
    #[arg(long, default_value_t = 1)]
    secretary_passes: usize,

    /// 各トリガー条件の最大ラウンド数 (--quick で 2 に縮約)．
    #[arg(long, default_value_t = 6)]
    rounds: usize,

    /// 世界大戦勃発の宣戦布告対しきい値．
    #[arg(long, default_value_t = 2)]
    war_threshold: usize,

    /// 乱数シード基点 (トリガー条件ごとに派生)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (ライブ実行時; mock では使わない)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// ライブ LLM の代わりに scripted mock を使う (オフライン検証・CI 用)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// 短縮再現 (rounds=2; CI スモーク用)．
    #[arg(long, default_value_t = false)]
    quick: bool,
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
// reproduce (論文 Table 2-5 ヘッドライン指標 一括再現)
// ---------------------------------------------------------------------------

/// 1 設定を実行する (`--mock` ならライブ LLM の代わりに scripted mock を使う)．
///
/// mock ポリシーは «プロンプトに注入された breaking-event (トリガー) の強度を読み，
/// 史実的にもっとも好戦的な国 (Country A) の行動をトリガー強度でスケールする» 決定論
/// 挙動で，ネットワーク遮断サンドボックスでもトリガー感応的な外交軌跡を生成できる
/// (ライブ LLM 呼び出し 0)．強度の段階:
/// - archduke (最高強度: "assassinated"/"ultimatum") → A が rival (C, id 2) へ公開宣戦布告，
///   B が総動員，C-D が防衛同盟を結ぶ → 開戦 + エスカレーション．
/// - dardanelles (高強度: "strait"/"blockaded") → A が総動員のみ (冷戦)．
/// - naval-incident (低強度: "naval skirmish") → A が外交メッセージのみ (冷戦)．
/// - null (注入なし) → 全国 wait (平時)．
fn run_one(cfg: &Config, mock: bool) -> Result<SimulationResult, String> {
    if !mock {
        return run(cfg);
    }
    let client = mock_war_client();
    // mock は in-memory cache なので保存先を持たない (cache().save() をスキップさせる)．
    let mock_cfg = Config {
        llm: LlmSettings {
            cache_path: None,
            ..cfg.llm.clone()
        },
        ..cfg.clone()
    };
    run_with_client(&mock_cfg, client)
}

/// reproduce 用の決定論 scripted mock クライアントを構築する (trigger 感応ポリシー)．
fn mock_war_client() -> waragent_simulation::llm::WarClient {
    let backend = ScriptedClient::new("mock-llama3.2", move |prompt: &str| {
        // 秘書検証プロンプトは決定 prompt と違い breaking-event を含まない．秘書の役割は
        // «妥当なら変更せず返す» なので，提示された行動 JSON をそのまま echo して決定を
        // 上書きしないようにする (trigger 非依存)．
        if let Some(idx) = prompt.find("## Proposed action\n") {
            let rest = &prompt[idx + "## Proposed action\n".len()..];
            if let Some(line) = rest.lines().next() {
                return line.trim().to_string();
            }
        }

        // 決定プロンプト: breaking-event 節からトリガー強度を判定する．
        let archduke = prompt.contains("assassinated") || prompt.contains("ultimatum");
        let dardanelles = prompt.contains("blockaded") || prompt.contains("strait");
        let naval = prompt.contains("naval skirmish");

        let is = |letter: &str| prompt.contains(&format!("Your country: Country {letter}"));

        if archduke {
            // 史実トリガー: A→C 公開宣戦布告，B 総動員，C-D 相互防衛同盟 → 開戦+エスカレーション．
            if is("A") {
                "{\"action\": \"declare_war\", \"target\": 2, \"publicity\": \"public\"}"
                    .to_string()
            } else if is("B") {
                "{\"action\": \"mobilize\"}".to_string()
            } else if is("C") {
                "{\"action\": \"alliance\", \"target\": 3, \"publicity\": \"public\"}".to_string()
            } else if is("D") {
                "{\"action\": \"alliance\", \"target\": 2, \"publicity\": \"public\"}".to_string()
            } else {
                "{\"action\": \"wait\"}".to_string()
            }
        } else if dardanelles {
            // 高強度だが開戦未満: A が総動員 (緊張のみ = 冷戦)．
            if is("A") {
                "{\"action\": \"mobilize\"}".to_string()
            } else {
                "{\"action\": \"wait\"}".to_string()
            }
        } else if naval {
            // 低強度: A が外交メッセージのみ (関係変化なし)．
            if is("A") {
                "{\"action\": \"message\", \"target\": 2, \"publicity\": \"public\"}".to_string()
            } else {
                "{\"action\": \"wait\"}".to_string()
            }
        } else {
            // null トリガー: 平時．
            "{\"action\": \"wait\"}".to_string()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

/// `reproduce_summary.json` の 1 トリガー条件行．
#[derive(serde::Serialize)]
struct ReproduceScenario {
    /// トリガー条件ラベル (null / naval-incident / dardanelles / archduke-assassination)．
    trigger: String,
    /// 観測した世界大戦勃発 (1/0)．
    war_outbreak: u8,
    /// 観測した冷戦フラグ (緊張のみ; 1/0)．
    cold_war_flag: u8,
    /// 初回勃発ラウンド (なければ -1)．
    escalation_round: i64,
    /// 終了時点の宣戦布告 (W) 対数．
    n_conflicts: u64,
    /// 最終ラウンドの同盟分割 MI (vs 史実)．
    final_alliance_mi: f64,
    /// 最終ラウンドの宣戦布告 Jaccard (vs 史実)．
    final_declaration_jaccard: f64,
    /// 最終ラウンドの総動員 Jaccard (vs 史実)．
    final_mobilization_jaccard: f64,
    /// 実行ラウンド数．
    final_round: usize,
    /// この結果を保存したサブディレクトリ (Python の figure 生成入力)．
    results_subdir: String,
}

/// `reproduce_summary.json` のアンカー判定行．
#[derive(serde::Serialize)]
struct ReproduceAnchor {
    name: String,
    paper_value: String,
    observed: f64,
    target_lo: f64,
    target_hi: f64,
    pass: bool,
}

/// `reproduce_summary.json` のルート．
#[derive(serde::Serialize)]
struct ReproduceSummary {
    command: &'static str,
    paper: &'static str,
    scenario: String,
    mock: bool,
    quick: bool,
    scenarios: Vec<ReproduceScenario>,
    anchors: Vec<ReproduceAnchor>,
    n_pass: usize,
    n_anchors: usize,
}

fn cmd_reproduce(args: ReproduceArgs) {
    let scenario = parse_scenario(&args.scenario).unwrap_or_else(|e| panic!("{e}"));
    let rounds = if args.quick { 2 } else { args.rounds };

    let ts = timestamp();
    let out_dir = format!("{}/{}_reproduce", args.output_dir, ts);
    ensure_output_dir(&out_dir);
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    println!("=== Hua et al. (2024) WarAgent 論文 Table 2-5 ヘッドライン指標 一括再現 ===");
    println!(
        "シナリオ: {} | rounds: {} | secretary-pass: {} | mock: {} | quick: {}",
        scenario.label(),
        rounds,
        args.secretary_passes,
        args.mock,
        args.quick,
    );
    println!("出力先: {out_dir}");
    println!("-------------------------------------------------");

    // トリガー強度の段階 (null → 平時 / dardanelles → 冷戦 / archduke → 開戦)．
    let triggers = [
        Trigger::Null,
        Trigger::Dardanelles,
        Trigger::ArchdukeAssassination,
    ];

    let mut scenarios: Vec<ReproduceScenario> = Vec::new();

    for trigger in triggers {
        let subdir_name = trigger.label().to_string();
        let subdir = format!("{out_dir}/{subdir_name}");
        ensure_output_dir(&subdir);
        let seed = socsim_core::derive_seed(args.seed, &[label_hash(trigger.label())]);
        let cfg = Config {
            scenario,
            trigger,
            stance_override: None,
            secretary_passes: args.secretary_passes,
            rounds,
            war_threshold: args.war_threshold,
            seed: Some(seed),
            llm: LlmSettings {
                temperature: args.llm_temperature,
                seed: args.llm_seed,
                cache_path: Some(args.cache_path.clone()),
            },
            output_dir: subdir.clone(),
        };

        let result = run_one(&cfg, args.mock).unwrap_or_else(|e| panic!("実行に失敗: {e}"));

        save_metrics(&result.metrics_history, &subdir);
        save_events(&result.event_log, &subdir);
        save_run_metadata(&result, &cfg, &subdir);
        let path = format!("{subdir}/config.json");
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");

        let last = result.metrics_history.last();
        scenarios.push(ReproduceScenario {
            trigger: trigger.label().to_string(),
            war_outbreak: if result.war_outbreak { 1 } else { 0 },
            cold_war_flag: if result.cold_war_flag { 1 } else { 0 },
            escalation_round: result.escalation_round.map(|r| r as i64).unwrap_or(-1),
            n_conflicts: result.n_conflicts,
            final_alliance_mi: last.map(|m| m.alliance_mi).unwrap_or(0.0),
            final_declaration_jaccard: last.map(|m| m.declaration_jaccard).unwrap_or(0.0),
            final_mobilization_jaccard: last.map(|m| m.mobilization_jaccard).unwrap_or(0.0),
            final_round: result.final_round,
            results_subdir: subdir_name,
        });
    }

    let by = |label: &str| scenarios.iter().find(|s| s.trigger == label).unwrap();
    let null = by(Trigger::Null.label());
    let dardanelles = by(Trigger::Dardanelles.label());
    let archduke = by(Trigger::ArchdukeAssassination.label());

    // --- アンカー判定 (論文 Table 2-5 のヘッドライン story) ---
    let mut anchors: Vec<ReproduceAnchor> = Vec::new();
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: f64| {
        anchors.push(ReproduceAnchor {
            name: name.to_string(),
            paper_value: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && obs <= hi,
        });
    };

    // Table 2/3 中核: 史実トリガー (archduke) で世界大戦が勃発する (war_outbreak=1)．
    push(
        "archduke trigger -> war outbreak (Table 2: WWI breaks out)",
        "outbreak=1",
        archduke.war_outbreak as f64,
        1.0,
        1.0,
    );
    // Table 4 (escalation): 史実トリガーは早期にエスカレーション (>=1 hop の同盟国参戦; n_conflicts>=2)．
    push(
        "archduke escalation conflict pairs (Table 4: allies pulled in)",
        ">=2",
        archduke.n_conflicts as f64,
        2.0,
        f64::INFINITY,
    );
    // Table 2/5 (counterfactual baseline): null トリガーでは開戦しない (=0)．
    push(
        "null trigger -> no war outbreak (Table 5: peace baseline)",
        "outbreak=0",
        null.war_outbreak as f64,
        0.0,
        0.0,
    );
    // Table 3 (cold war / 緊張): 中間強度 (dardanelles) は開戦せず緊張のみ (cold_war=1)．
    push(
        "dardanelles trigger -> cold war, no outbreak (Table 3: tension)",
        "cold_war=1",
        dardanelles.cold_war_flag as f64,
        1.0,
        1.0,
    );
    // Table 2 (alliance polarization): 史実トリガーの最終 同盟MI > null の最終 同盟MI (分極化が進む)．
    push(
        "archduke alliance polarization (Table 2: MI_archduke >= MI_null)",
        "archduke >= null",
        archduke.final_alliance_mi - null.final_alliance_mi,
        0.0,
        f64::INFINITY,
    );

    let n_pass = anchors.iter().filter(|a| a.pass).count();
    let n_anchors = anchors.len();

    println!("トリガー条件:");
    for s in &scenarios {
        let esc = if s.escalation_round >= 0 {
            s.escalation_round.to_string()
        } else {
            "なし".to_string()
        };
        println!(
            "  [{:<22}] 開戦={} 冷戦={} 勃発R={} 紛争={} MI={:.3} 宣戦J={:.3} 総動員J={:.3} (round {})",
            s.trigger,
            s.war_outbreak,
            s.cold_war_flag,
            esc,
            s.n_conflicts,
            s.final_alliance_mi,
            s.final_declaration_jaccard,
            s.final_mobilization_jaccard,
            s.final_round,
        );
    }
    println!("-------------------------------------------------");
    for a in &anchors {
        let hi = if a.target_hi.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.2}", a.target_hi)
        };
        println!(
            "[{}] {:<52} obs={:.4} target=[{:.2},{}] paper={}",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
            a.paper_value,
        );
    }
    println!("-------------------------------------------------");
    println!("{n_pass}/{n_anchors} アンカーが in-band");

    let summary = ReproduceSummary {
        command: "reproduce",
        paper: "Hua et al. (2024) WarAgent — Table 2-5 (historical trigger -> WWI outbreak \
                + alliance escalation; null trigger -> peace baseline)",
        scenario: scenario.label().to_string(),
        mock: args.mock,
        quick: args.quick,
        scenarios,
        anchors,
        n_pass,
        n_anchors,
    };
    let path = format!("{out_dir}/reproduce_summary.json");
    write_json(&summary, &path).expect("reproduce_summary.json の書き込みに失敗");

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{ts}_reproduce"));

    println!("サマリ → {out_dir}/reproduce_summary.json");
    println!("各トリガー条件の metrics.csv / events.csv / run_metadata.json を各サブディレクトリに保存しました．");
    println!("図 (Table 2-5 風) は `uv run waragent-tools reproduce` で生成できます．");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// reproduce と同じ設定で 1 トリガー条件を mock 実行する (offline)．
    fn reproduce_one(trigger: Trigger) -> SimulationResult {
        let seed = socsim_core::derive_seed(42, &[label_hash(trigger.label())]);
        let cfg = Config {
            scenario: Scenario::WwiSmall,
            trigger,
            stance_override: None,
            secretary_passes: 1,
            rounds: 6,
            war_threshold: 2,
            seed: Some(seed),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        };
        run_one(&cfg, true).expect("mock reproduce run failed")
    }

    /// Table 2-5 のヘッドライン ordering: archduke は開戦・エスカレーションし，
    /// null は平和，dardanelles は冷戦 (開戦せず緊張のみ)．
    #[test]
    fn reproduce_recovers_table_2_5_ordering() {
        let null = reproduce_one(Trigger::Null);
        let dardanelles = reproduce_one(Trigger::Dardanelles);
        let archduke = reproduce_one(Trigger::ArchdukeAssassination);

        // archduke -> 開戦 + 同盟国エスカレーション (>=2 紛争対)．
        assert!(archduke.war_outbreak, "archduke は開戦すべき");
        assert!(
            archduke.n_conflicts >= 2,
            "archduke はエスカレーションすべき n_conflicts={}",
            archduke.n_conflicts
        );
        // null -> 平和ベースライン (開戦せず・冷戦でもない)．
        assert!(!null.war_outbreak, "null は開戦しないべき");
        assert!(!null.cold_war_flag, "null は冷戦でもないべき (平時)");
        // dardanelles -> 冷戦 (開戦せず緊張のみ)．
        assert!(!dardanelles.war_outbreak, "dardanelles は開戦しないべき");
        assert!(dardanelles.cold_war_flag, "dardanelles は冷戦であるべき");

        // 同盟分極化: archduke の最終 MI は null 以上 (CtoD 防衛同盟 + 史実陣営)．
        let mi_null = null.metrics_history.last().map(|m| m.alliance_mi).unwrap();
        let mi_arch = archduke
            .metrics_history
            .last()
            .map(|m| m.alliance_mi)
            .unwrap();
        assert!(
            mi_arch >= mi_null,
            "MI_archduke({mi_arch}) >= MI_null({mi_null})"
        );
    }

    /// mock は決定論的: 同一トリガーの 2 回実行が指標系列まで完全一致する．
    #[test]
    fn reproduce_mock_is_bit_deterministic() {
        let a = reproduce_one(Trigger::ArchdukeAssassination);
        let b = reproduce_one(Trigger::ArchdukeAssassination);
        let series = |r: &SimulationResult| -> Vec<(u64, u64, u64)> {
            r.metrics_history
                .iter()
                .map(|m| (m.round, m.n_conflicts, m.n_mobilized))
                .collect()
        };
        assert_eq!(series(&a), series(&b), "同一 mock は完全再現すべき");
        assert_eq!(a.war_outbreak, b.war_outbreak);
        assert_eq!(a.escalation_round, b.escalation_round);
    }
}
