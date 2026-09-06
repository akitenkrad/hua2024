//! Hua et al. (2024) "War and Peace (WarAgent)" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で LLM 駆動の国エージェント外交 ABM を実行する．
//! `sweep`     : トリガー強度 × スタンス を走査する．親 run 1 本 + セルごとの子 run．
//! `reproduce` : 論文 (Hua et al. 2024) の Table 2-5 ヘッドライン指標 — トリガー強度に
//!               応じた開戦頻度・エスカレーション・同盟分極化 — を 3 つのトリガー条件
//!               (null / dardanelles / archduke) を `wwi-small` で走らせて確かめる．
//!               親 run 1 本 + トリガー条件ごとの子 run で，条件をまたいだ差は親の
//!               sweep スコープ指標に入る．`--mock` でライブ LLM 無しに決定論再現する．
//!
//! 反実仮想分析・WWII/戦国時代シナリオ・脱匿名化比較は本コマンドの対象外 (拡張点)．
//!
//! サブコマンド 1 回が runvault の run 1 本になる (掃引は親 1 本 + 子)．出力の
//! 置き場と同一性 (run ディレクトリ・`config.json`・`metrics.csv`・`events.jsonl`)
//! は runvault が持つので，ここではタイムスタンプ付きディレクトリも `latest`
//! symlink も作らない．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};
use waragent_simulation::config::{
    derive_run_seed, parse_scenario, parse_stance, parse_trigger, scenario_country_count, Config,
    LlmSettings, Scenario, Trigger,
};
use waragent_simulation::llm::{build_live_client, wrap_client, WarClient};
use waragent_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID, SWEEP_SCOPE};
use waragent_simulation::simulation::{run_with_client_observed, SimulationResult};
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

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で LLM 駆動の国エージェント外交 ABM を実行する．
    Run(RunArgs),
    /// トリガー強度 × スタンス を走査し，開戦率・同盟 MI を集計する．
    Sweep(SweepArgs),
    /// 論文 Table 2-5 のヘッドライン指標を一括再現する (トリガー条件ごとに子 run)．
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

    /// 乱数シード (省略時は 42; socsim コア層のみ支配)．
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

    /// 結果出力ベースディレクトリ (runvault の results root)．
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

    /// 結果出力ベースディレクトリ (runvault の results root)．
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

    /// 結果出力ベースディレクトリ (runvault の results root)．
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

/// `sweep` 親 run の `parameters`．掃引の格子そのものを持つ．
#[derive(serde::Serialize)]
struct SweepConfigJson {
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

/// `reproduce` 親 run の `parameters`．トリガー条件と共通設定を持つ．
///
/// `rounds` は `--quick` 適用後の実効値である (`--quick` は `rounds` を 2 に
/// 縮約するので，フラグではなく効いた値を残す)．`mock` は結果を決める値なので
/// 持つ — 子は `llm` ブロックの endpoint (`mock://…`) でも判別できるが，親は
/// 自分では LLM を呼ばないので `llm` ブロックを持たない．
#[derive(serde::Serialize)]
struct ReproduceConfigJson {
    scenario: String,
    trigger_values: Vec<String>,
    secretary_passes: usize,
    rounds: usize,
    war_threshold: usize,
    seed: u64,
    mock: bool,
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

/// LLM レイヤ設定を組む．mock は永続キャッシュを持たないので `cache_path` を落とす．
fn llm_settings(temperature: f32, seed: u64, cache_path: &str, mock: bool) -> LlmSettings {
    LlmSettings {
        temperature,
        seed,
        cache_path: (!mock).then(|| cache_path.to_string()),
    }
}

/// LLM キャッシュの置き場を用意する (ライブ実行のみ; mock は in-memory)．
fn ensure_cache_dir(cfg: &Config) {
    if let Some(parent) = cfg
        .llm
        .cache_path
        .as_deref()
        .and_then(|path| Path::new(path).parent())
    {
        let _ = fs::create_dir_all(parent);
    }
}

/// LLM クライアントを 1 本組む．
///
/// `run.json` の `llm` ブロックに書くモデル名と endpoint は，実際に応答する
/// バックエンドから採らないと意味を持たないので，組み立ては `Run::start` より前に
/// 置く (このために `simulation::run` を消してある — 中でクライアントを組む入口が
/// 残っていると，`llm` ブロックを埋めないまま記録できてしまう)．
fn build_client(cfg: &Config, mock: bool) -> WarClient {
    if mock {
        mock_war_client()
    } else {
        build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"))
    }
}

/// `RunOptions` の共通部分 (シミュレーション 1 本ぶんの子/単独 run)．
fn run_options(cfg: &Config, output_dir: &str, seed: u64, client: &WarClient) -> RunOptions {
    let parameters = cfg.to_run_config_json();
    RunOptions::new(EXPERIMENT, "run")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(output_dir)
        .parameters(&parameters)
        .expect("runvault: parameters の組み立てに失敗")
        .seed_pointers(["/seed"])
        .master_seed(seed)
        .llm(record::llm_block(
            client.inner().model(),
            client.inner().endpoint(),
            cfg.llm.temperature,
        ))
        .replication(record::replication())
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

    let base_seed = args.seed.unwrap_or(42);
    let runs = args.runs.max(1);

    // 記録するのは最後の 1 本．`--runs N` は同じ条件を N 本回して最後の試行の詳細
    // だけを残す既存の動きなので (旧実装も `save_metrics` を最終試行でしか呼んで
    // いない)，`master_seed` には実際に世界を支配した `derive_run_seed(base, N-1)`
    // を書き，`replicate_index` を N-1 にする．CLI で与えた根のシードは
    // `/parameters.seed` にあり，seed_pointers 経由で execution_hash に残る．
    let recorded_seed = derive_run_seed(base_seed, runs - 1);

    let base_cfg = Config {
        scenario,
        trigger,
        stance_override,
        secretary_passes: args.secretary_passes,
        rounds: args.rounds,
        war_threshold: args.war_threshold,
        // `parameters` に載るのは CLI で与えた根のシード．実際に世界を支配した
        // 派生シードは `master_seed` が持つ．
        seed: Some(base_seed),
        llm: llm_settings(args.llm_temperature, args.llm_seed, &args.cache_path, false),
    };
    ensure_cache_dir(&base_cfg);

    // クライアントは run を開始する前に組む (`llm` ブロックのため)．最初の 1 本で
    // そのまま使い，2 本目以降は旧実装と同じく 1 本ごとに組み直す．
    let mut pending = Some(build_client(&base_cfg, false));
    let mut rv = Run::start(
        run_options(
            &base_cfg,
            &args.output_dir,
            recorded_seed,
            pending.as_ref().expect("直前に組んだクライアント"),
        )
        .replicate_index((runs - 1) as u64),
    )
    .expect("runvault: run の開始に失敗");
    // 論文 Table 2 の報告値は，この 3 指標をその条件で直接測る `run` にだけ置く．
    record::log_paper_references(&mut rv);

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
        runs,
    );
    println!(
        "LLM: temp={} llm_seed={} cache={} | seed: {}",
        args.llm_temperature, args.llm_seed, args.cache_path, base_seed
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    // 進捗の 1 単位は 1 ラウンド．費用がそこにあり，1 ラウンドは全国にその
    // ラウンドの行動を尋ね，さらに秘書検証を secretary_passes 回かける —
    // どれもモデル呼び出しである．試行を単位にすると，ライブの 1 本は 0/1 と
    // 出したきり終わりまで黙る．
    //
    // 分母は持たない．`BoardUpdateMechanism` は世界大戦が勃発した時点で
    // `request_stop()` するので，`rounds` は «上限» であって仕事の量ではない．
    // mock の実測では既定の archduke-assassination トリガーが 6 ラウンド中
    // 1 ラウンド目で止まり，null と dardanelles は 6 ラウンド回った — 同じ
    // `--rounds 6` に対して 6 倍の開きがある．何ラウンド目で止まるかは
    // 走らせるまで分からないので，上限を分母に置けば見積もりは «自信をもって
    // 外れた» ものになる．
    let mut stage = rv.unbounded_stage("rounds");
    let mut last_result: Option<SimulationResult> = None;
    let mut outbreak_count = 0usize;
    let mut cold_war_count = 0usize;

    for run_idx in 0..runs {
        let seed = derive_run_seed(base_seed, run_idx);
        let cfg = Config {
            seed: Some(seed),
            ..base_cfg.clone()
        };

        let client = pending.take().unwrap_or_else(|| build_client(&cfg, false));
        let result = run_with_client_observed(&cfg, client, |_| stage.tick())
            .unwrap_or_else(|e| panic!("実行に失敗: {e}"));
        if result.war_outbreak {
            outbreak_count += 1;
        }
        if result.cold_war_flag {
            cold_war_count += 1;
        }

        // 最後の試行の詳細を記録する (代表 run)．
        if run_idx + 1 == runs {
            record::log_simulation(&mut rv, &result, scenario_country_count(scenario));
            last_result = Some(result);
        }
    }

    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

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

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("メトリクス → {}/metrics.csv", dir.display());
    println!("行動ログ   → {}/events.jsonl", dir.display());
    println!("論文値     → {}/reference.csv", dir.display());
    println!("設定       → {}/config.json", dir.display());
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

    let n_total = triggers.len() * stances.len() * args.runs;

    // 親 run: 格子の定義そのものを parameters に持つ．個別セルの指標は書かない．
    // 親は 1 本のシミュレーションではないので master_seed を名乗らない (セルごとの
    // 子が派生シードをそれぞれ持つ)．base seed は /parameters.seed と seed_pointers
    // 経由で execution_hash に残る．sweep_id は runvault が親の run_slug で埋める．
    let sweep_parameters = SweepConfigJson {
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
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Hua et al. (2024) WarAgent 感度分析 (トリガー × スタンス) ===");
    println!(
        "シナリオ: {} | トリガー {} 種 × スタンス {} 種 | 試行: {} | 合計: {} 実行",
        scenario.label(),
        triggers.len(),
        stances.len(),
        args.runs,
        n_total,
    );
    println!("出力先: {}", parent.dir().display());
    println!("-----------------------------------------------------------");

    // コンソールの要約に使うだけの控え (ディスクには書かない; 同じ値は子 run の
    // 指標にある)．
    let mut console: Vec<(&'static str, bool, f64)> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    // 掃引全体で stage を 1 つ．トリガーごとに開け直すと小さな tally が 3 つ並ぶ
    // だけで，掃引全体の進み方が読めなくなる．勃発で早期に止まるかどうかは
    // トリガーで決まるため 1 試行の長さは条件で変わるが，stage は分母を持たない
    // ので守るべき割合も見積もりも無く，分ける理由にならない．
    let mut stage = parent.unbounded_stage("rounds");

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
                    llm: llm_settings(args.llm_temperature, args.llm_seed, &args.cache_path, false),
                };
                ensure_cache_dir(&cfg);
                let client = build_client(&cfg, false);

                // 子は «そのセルの run» そのもの．master_seed は base から派生した
                // 実際に使われるシードで，同一セルの繰り返しは replicate_index で
                // 分ける．parameters は手で回した `run` と同じ形なので，同じ条件
                // なら config_hash が一致する．
                let mut child = Run::start(
                    run_options(&cfg, &args.output_dir, seed, &client)
                        .replicate_index(run_idx as u64)
                        .lineage(Lineage {
                            sweep_id: Some(sweep_id.clone()),
                            parent_run_uid: Some(parent_run_uid.clone()),
                            ..Default::default()
                        }),
                )
                .expect("runvault: 子 run の開始に失敗");

                let result = run_with_client_observed(&cfg, client, |_| stage.tick())
                    .unwrap_or_else(|e| panic!("実行に失敗: {e}"));
                record::log_simulation(&mut child, &result, scenario_country_count(scenario));
                child.finish().expect("runvault: 子 run の完了に失敗");

                let final_mi = result
                    .metrics_history
                    .last()
                    .map(|m| m.alliance_mi)
                    .unwrap_or(0.0);
                console.push((trigger.label(), result.war_outbreak, final_mi));
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

    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");

    println!("===========================================================");
    println!("スイープ完了: {n_total} 実行");
    println!("-----------------------------------------------------------");
    println!("トリガー別の開戦発生頻度 / 平均 同盟MI:");
    for &trigger in &triggers {
        let rows: Vec<&(&str, bool, f64)> =
            console.iter().filter(|r| r.0 == trigger.label()).collect();
        if rows.is_empty() {
            continue;
        }
        let outbreak_freq = rows.iter().filter(|r| r.1).count() as f64 / rows.len() as f64;
        let avg_mi = rows.iter().map(|r| r.2).sum::<f64>() / rows.len() as f64;
        println!(
            "  trigger={} → 開戦 = {:.1}% | 同盟MI = {:.3}",
            trigger.label(),
            outbreak_freq * 100.0,
            avg_mi
        );
    }
    println!("-----------------------------------------------------------");
    println!("親 run → {}", dir.display());
    println!("子 run は lineage.parent_run_uid で親を指す．");
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

// ---------------------------------------------------------------------------
// reproduce (論文 Table 2-5 ヘッドライン指標 一括再現)
// ---------------------------------------------------------------------------

/// reproduce 用の決定論 scripted mock クライアントを構築する (trigger 感応ポリシー)．
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
fn mock_war_client() -> WarClient {
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

/// 1 トリガー条件ぶんの観測 (コンソールの表とアンカー判定に使う控え)．
///
/// ディスクには «この構造体» としては書かない — 中身はすべて子 run の指標にある．
struct TriggerOutcome {
    trigger: &'static str,
    war_outbreak: u8,
    cold_war_flag: u8,
    escalation_round: Option<u64>,
    n_conflicts: u64,
    final_alliance_mi: f64,
    final_declaration_jaccard: f64,
    final_mobilization_jaccard: f64,
    final_round: usize,
}

/// この再現実装が置いたアンカー (論文の定性的な主張を帯に落としたもの)．
///
/// 帯も PASS/OFF も論文が印字した数ではないので記録しない — コンソールに残す．
/// 観測値そのものは子 run (と親の sweep スコープ指標) にある．
struct Anchor {
    name: &'static str,
    paper_value: &'static str,
    observed: f64,
    target_lo: f64,
    target_hi: f64,
}

impl Anchor {
    fn pass(&self) -> bool {
        self.observed >= self.target_lo && self.observed <= self.target_hi
    }
}

fn cmd_reproduce(args: ReproduceArgs) {
    let scenario = parse_scenario(&args.scenario).unwrap_or_else(|e| panic!("{e}"));
    let rounds = if args.quick { 2 } else { args.rounds };

    // トリガー強度の段階 (null → 平時 / dardanelles → 冷戦 / archduke → 開戦)．
    let triggers = [
        Trigger::Null,
        Trigger::Dardanelles,
        Trigger::ArchdukeAssassination,
    ];

    // 親 run: トリガー条件と共通設定を parameters に持ち，条件をまたいだ差
    // (同盟分極化のギャップ) を sweep スコープの指標として書く．条件ごとの値は
    // それぞれの子 run にあるので，親には重ねない．
    let parent_parameters = ReproduceConfigJson {
        scenario: scenario.label().to_string(),
        trigger_values: triggers.iter().map(|t| t.label().to_string()).collect(),
        secretary_passes: args.secretary_passes,
        rounds,
        war_threshold: args.war_threshold,
        seed: args.seed,
        mock: args.mock,
        llm_temperature: args.llm_temperature,
        llm_seed: args.llm_seed,
    };
    let mut parent = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parent_parameters)
            .expect("runvault: reproduce の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: reproduce 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: reproduce 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Hua et al. (2024) WarAgent 論文 Table 2-5 ヘッドライン指標 一括再現 ===");
    println!(
        "シナリオ: {} | rounds: {} | secretary-pass: {} | mock: {} | quick: {}",
        scenario.label(),
        rounds,
        args.secretary_passes,
        args.mock,
        args.quick,
    );
    println!("出力先: {}", parent.dir().display());
    println!("-------------------------------------------------");

    let mut outcomes: Vec<TriggerOutcome> = Vec::new();

    // 3 トリガーを通して stage は 1 つ，親 run に開ける．トリガー条件は子 run に
    // 分かれているが，reproduce 全体の進み方は 1 つのログで読めたほうがよい．
    // トリガーごとに開け直すと小さな tally が 3 つ並ぶだけになる (理由は sweep と
    // 同じ)．
    let mut stage = parent.unbounded_stage("rounds");

    for trigger in triggers {
        let seed = socsim_core::derive_seed(args.seed, &[label_hash(trigger.label())]);
        let cfg = Config {
            scenario,
            trigger,
            stance_override: None,
            secretary_passes: args.secretary_passes,
            rounds,
            war_threshold: args.war_threshold,
            seed: Some(seed),
            llm: llm_settings(
                args.llm_temperature,
                args.llm_seed,
                &args.cache_path,
                args.mock,
            ),
        };
        ensure_cache_dir(&cfg);
        let client = build_client(&cfg, args.mock);

        let mut child = Run::start(
            run_options(&cfg, &args.output_dir, seed, &client)
                .replicate_index(0)
                .lineage(Lineage {
                    sweep_id: Some(sweep_id.clone()),
                    parent_run_uid: Some(parent_run_uid.clone()),
                    ..Default::default()
                }),
        )
        .expect("runvault: 子 run の開始に失敗");

        let result = run_with_client_observed(&cfg, client, |_| stage.tick())
            .unwrap_or_else(|e| panic!("実行に失敗: {e}"));
        record::log_simulation(&mut child, &result, scenario_country_count(scenario));
        child.finish().expect("runvault: 子 run の完了に失敗");

        let last = result.metrics_history.last();
        outcomes.push(TriggerOutcome {
            trigger: trigger.label(),
            war_outbreak: if result.war_outbreak { 1 } else { 0 },
            cold_war_flag: if result.cold_war_flag { 1 } else { 0 },
            escalation_round: result.escalation_round,
            n_conflicts: result.n_conflicts,
            final_alliance_mi: last.map(|m| m.alliance_mi).unwrap_or(0.0),
            final_declaration_jaccard: last.map(|m| m.declaration_jaccard).unwrap_or(0.0),
            final_mobilization_jaccard: last.map(|m| m.mobilization_jaccard).unwrap_or(0.0),
            final_round: result.final_round,
        });
    }

    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    let by = |label: &str| {
        outcomes
            .iter()
            .find(|s| s.trigger == label)
            .expect("トリガー条件の結果が無い")
    };
    let null = by(Trigger::Null.label());
    let dardanelles = by(Trigger::Dardanelles.label());
    let archduke = by(Trigger::ArchdukeAssassination.label());

    // 親が持つのは «条件をまたいだ» 量だけである．同盟分極化のギャップ
    // (史実トリガー − null トリガー) は 1 本の run では測れない．
    let mi_gap = archduke.final_alliance_mi - null.final_alliance_mi;
    parent
        .log_metrics(
            SWEEP_SCOPE,
            &[("alliance_mi_gap_archduke_minus_null", mi_gap)],
        )
        .expect("reproduce 親の集約指標の記録に失敗");

    // --- アンカー判定 (論文 Table 2-5 のヘッドライン story) ---
    let anchors = [
        // Table 2/3 中核: 史実トリガー (archduke) で世界大戦が勃発する (war_outbreak=1)．
        Anchor {
            name: "archduke trigger -> war outbreak (Table 2: WWI breaks out)",
            paper_value: "outbreak=1",
            observed: archduke.war_outbreak as f64,
            target_lo: 1.0,
            target_hi: 1.0,
        },
        // Table 4 (escalation): 史実トリガーは早期にエスカレーションする
        // (>=1 hop の同盟国参戦; n_conflicts>=2)．
        Anchor {
            name: "archduke escalation conflict pairs (Table 4: allies pulled in)",
            paper_value: ">=2",
            observed: archduke.n_conflicts as f64,
            target_lo: 2.0,
            target_hi: f64::INFINITY,
        },
        // Table 2/5 (counterfactual baseline): null トリガーでは開戦しない (=0)．
        Anchor {
            name: "null trigger -> no war outbreak (Table 5: peace baseline)",
            paper_value: "outbreak=0",
            observed: null.war_outbreak as f64,
            target_lo: 0.0,
            target_hi: 0.0,
        },
        // Table 3 (cold war / 緊張): 中間強度 (dardanelles) は開戦せず緊張のみ．
        Anchor {
            name: "dardanelles trigger -> cold war, no outbreak (Table 3: tension)",
            paper_value: "cold_war=1",
            observed: dardanelles.cold_war_flag as f64,
            target_lo: 1.0,
            target_hi: 1.0,
        },
        // Table 2 (alliance polarization): 史実トリガーの最終 同盟MI > null の最終 同盟MI．
        Anchor {
            name: "archduke alliance polarization (Table 2: MI_archduke >= MI_null)",
            paper_value: "archduke >= null",
            observed: mi_gap,
            target_lo: 0.0,
            target_hi: f64::INFINITY,
        },
    ];

    let n_pass = anchors.iter().filter(|a| a.pass()).count();
    let n_anchors = anchors.len();

    println!("トリガー条件:");
    for s in &outcomes {
        let esc = match s.escalation_round {
            Some(r) => r.to_string(),
            None => "なし".to_string(),
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
    // 帯は論文の主張ではなくこの再現実装が置いたものなので，記録せず表示だけする．
    for a in &anchors {
        let hi = if a.target_hi.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.2}", a.target_hi)
        };
        println!(
            "[{}] {:<52} obs={:.4} target=[{:.2},{}] paper={}",
            if a.pass() { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
            a.paper_value,
        );
    }
    println!("-------------------------------------------------");
    println!("{n_pass}/{n_anchors} アンカーが in-band");

    let dir = parent
        .finish()
        .expect("runvault: reproduce 親 run の完了に失敗");
    println!("集約     → {}/metrics.csv (scope=sweep)", dir.display());
    println!("トリガー条件ごとの実行は lineage.parent_run_uid で親を指す子 run にある．");
    println!("図 (Table 2-5 風) は `uv run waragent-tools reproduce` で生成できます．");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // 試験は移行前と同じ入口を呼ぶ (コールバック無しの薄い包み)．本体は
    // これを使わないので，import はここに置く．
    use waragent_simulation::simulation::run_with_client;

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
        };
        run_with_client(&cfg, mock_war_client()).expect("mock reproduce run failed")
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

    /// 掃引セルのシードは (base, trigger, stance, index) が同じなら常に同じ値になる．
    /// この性質が壊れると，記録した master_seed から run を組み直せなくなる．
    #[test]
    fn sweep_seeds_are_reproducible_and_distinct() {
        let base = sweep_seed(42, &Trigger::Null, &Stance::Conservative, 0);
        assert_eq!(
            base,
            sweep_seed(42, &Trigger::Null, &Stance::Conservative, 0)
        );
        assert_ne!(
            base,
            sweep_seed(43, &Trigger::Null, &Stance::Conservative, 0),
            "base が効いていない"
        );
        assert_ne!(
            base,
            sweep_seed(42, &Trigger::Dardanelles, &Stance::Conservative, 0),
            "trigger が効いていない"
        );
        assert_ne!(
            base,
            sweep_seed(42, &Trigger::Null, &Stance::Aggressive, 0),
            "stance が効いていない"
        );
        assert_ne!(
            base,
            sweep_seed(42, &Trigger::Null, &Stance::Conservative, 1),
            "index が効いていない"
        );
    }
}
