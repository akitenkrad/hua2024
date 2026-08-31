//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` でも `sweep` / `reproduce` の子でも同一なので，
//! ここ 1 箇所で組み立てる．ラウンドごとの指標，run 全体を 1 つの値で表す指標，
//! 国の行動ログ，論文が報告した値の書き方もここに集める．
//!
//! # 行動ログをどこに置くか
//!
//! 旧 `events.csv` は `round,actor,action,target,publicity` で，`action` /
//! `publicity` はラベル (`declare_war` / `alliance` / `wait` …) である．非数値は
//! 指標にできないので `metrics.csv` には置けない．そのうえ 1 ラウンドに同じ国が
//! 2 行を持つことがある (`archduke` の round 0 では Country D が `alliance` と
//! `escalate_join_war` を続けて起こす) ので，«1 主体 1 時点 1 行» のコア語彙
//! `observation` の形にも収まらない．これは状態の観測ではなく行動ログなので，
//! 実験固有の名前空間つきイベント [`ACTION_EVENT`] に置く．
//!
//! 大きさは «国数 × ラウンド数 + エスカレーション参戦» で，論文規模 (WWI 8 カ国 ×
//! 6 ラウンド) でも 50 行程度・10 KB 未満に収まる (実測: `wwi` × 6 ラウンド ×
//! 秘書 1 パスで 48 行 8.8 KB)．間引く必要はない．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::metrics::RoundMetric;
use crate::simulation::SimulationResult;
use crate::world::Event;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
/// バイナリ名 (`waragent`) と揃える．
pub const EXPERIMENT: &str = "waragent";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "hua2024";
/// 分野．活性化順 (`RandomActivationScheduler`) と世界初期化が乱数駆動で
/// `master_seed` が要るので `simulation`．意思決定は LLM が担うが，測っているのは
/// モデルの安全性ではなく国際関係の動学なので `llm-safety` ではない．LLM 側の
/// 同一性は `run.json` の `llm` ブロックが持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// このモデルの 1 刻みは «外交の 1 ラウンド» — 各国が状況を読み，行動を決め，
/// 秘書が検証し，Board が更新されるまでの 1 巡である．語彙にそのまま `round` が
/// あるので，同じく LLM 駆動で 1 刻み = 1 巡の zhao2024 / gao2023 と揃える．
const T_UNIT: &str = "round";

/// ラウンドごと・run 全体の指標の粒度．いずれも «世界» 全体の集約なので `run`．
const SCOPE: &str = "run";

/// `reproduce` の親が持つ «条件をまたいだ集約» の粒度．
pub const SWEEP_SCOPE: &str = "sweep";

/// 国の行動ログの種別 (実験固有の名前空間つきイベント)．
pub const ACTION_EVENT: &str = "x.hua2024.action";

/// この再現実験が対象としている論文．
///
/// `run` も `sweep` / `reproduce` の子も同じ対象を持つ — 掃引はトリガー強度と
/// スタンスを変えて開戦の条件を見るためのもので，別の論文を相手にしてはいない．
pub fn replication() -> Replication {
    let mut work = Work::arxiv("2311.17227")
        .title(
            "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of \
             World Wars",
        )
        .year(2024)
        .source_version("arxiv-v2");
    // vault 側の同定にも使えるよう paper-id も残す (work_id は arXiv 側)．
    work.paper_id = Some("P00001798".to_string());
    Replication::new(work)
        .target(Target::table("table2", "Table 2"))
        .target(Target::claim(
            "trigger-intensity-drives-outbreak",
            "A stronger breaking event drives the countries from peace through cold-war tension \
             into an all-out war that pulls the allies in",
        ))
        .target(Target::claim(
            "alliance-polarization",
            "The historical trigger polarizes the alliance structure toward the historical \
             partition",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/hua2024/設計書.md")
}

// --------------------------------------------------------------------------- //
// LLM ブロック
// --------------------------------------------------------------------------- //

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める．推測しているのは分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.2:latest` のような動くエイリアスであることが
/// 多い．socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // 各国のプロンプトはプロフィール・Board・inbox から相手ごとに組み立てられ，
        // 固定の system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// --------------------------------------------------------------------------- //
// シミュレーション 1 本ぶんの記録
// --------------------------------------------------------------------------- //

/// シミュレーション 1 本ぶんを run へ書く (`run` サブコマンドと掃引の子で共通)．
///
/// `n_countries` は設定から決まる国数で，行動ログを持つ主体の数でもある
/// (`x.hua2024.action` は毎ラウンド全国ぶん書かれる)．
pub fn log_simulation(run: &mut Run, result: &SimulationResult, n_countries: usize) {
    log_rounds(run, &result.metrics_history);
    log_actions(run, &result.event_log);
    log_run_scope(run, result, n_countries);
}

/// ラウンドごとの指標を `metrics.csv` に書く．
///
/// 旧 `metrics.csv` の `round` 列が時間軸そのものになり，残りの 7 列がそのままの
/// 名前で `scope=run` のステップ指標になる．`war_outbreak` は 0/1 だがカテゴリに
/// 番号を振ったのではなく «そのラウンドまでに勃発したか» の指標変数で，ラウンドを
/// またいで 0 → 1 に動く (実測: `archduke` は round 0 で 1，`null` /
/// `dardanelles` は最後まで 0)．複数 run にわたる平均が開戦頻度になる．
fn log_rounds(run: &mut Run, history: &[RoundMetric]) {
    for m in history {
        run.log_metrics_at(
            m.round,
            T_UNIT,
            SCOPE,
            &[
                ("alliance_mi", m.alliance_mi),
                ("declaration_jaccard", m.declaration_jaccard),
                ("mobilization_jaccard", m.mobilization_jaccard),
                ("n_conflicts", m.n_conflicts as f64),
                ("n_mobilized", m.n_mobilized as f64),
                ("n_alliance_clusters", m.n_alliance_clusters as f64),
                ("war_outbreak", m.war_outbreak as f64),
            ],
        )
        .unwrap_or_else(|e| panic!("round {} の指標の記録に失敗: {e}", m.round));
    }
}

/// `events.jsonl` に書く行動 1 件．
///
/// 旧 `events.csv` の 5 列がそのまま欄になる．`unit_id` は行動主体の国で，
/// 同じラウンドに同じ国が 2 行持つことがある (`escalate_join_war`)．
///
/// 欄の名前は掃引パラメータ (`scenario` / `trigger` / `stance_override` /
/// `secretary_passes` / `rounds` / `war_threshold` / `n_countries` / `seed` /
/// `llm_temperature` / `llm_seed`) と重ならないようにしてある．
/// `runvault.read.sweep_events_table` は同名のパラメータ列でイベント列を上書き
/// するので，衝突すると黙って消える．
#[derive(Serialize)]
struct ActionEvent<'a> {
    unit_id: String,
    t: u64,
    t_unit: &'static str,
    actor: u64,
    action: &'a str,
    target: Option<u64>,
    publicity: &'a str,
}

/// 行動ログを 1 件 1 行で書く．
fn log_actions(run: &mut Run, events: &[Event]) {
    for e in events {
        let event = ActionEvent {
            unit_id: unit_id(e.actor),
            t: e.round,
            t_unit: T_UNIT,
            actor: e.actor,
            action: &e.action,
            target: e.target,
            publicity: &e.publicity,
        };
        run.log_event(ACTION_EVENT, &event).unwrap_or_else(|err| {
            panic!(
                "round {} の国 {} の行動 ({}) の記録に失敗: {err}",
                e.round, e.actor, e.action
            )
        });
    }
}

/// run 全体を 1 つの値で表す指標．
///
/// `n_units` は予約指標名で «観測主体の数» — このモデルでは行動ログを持つ国の数
/// である．実行時間は `status.json` の `duration_sec` が正本なので指標にはしない．
///
/// `war_outbreak` と `n_conflicts` はここには書かない．どちらも最終ラウンドの
/// ステップ指標と同じ数で (`SimulationResult` はその行から採っている)，同じ数を
/// 2 箇所に置くと食い違う余地ができる．`cold_war_flag` は履歴全体と初期同盟
/// クラスタ数から決まる別の量なので run スコープに置く．
///
/// `escalation_round` は勃発しなかった run では «無い» ので，行そのものを書かない
/// (旧 `run_metadata.json` は `null`，旧 `reproduce_summary.json` は -1 で埋めて
/// いた)．欠測を 0 で埋めない．
fn log_run_scope(run: &mut Run, result: &SimulationResult, n_countries: usize) {
    let calls = result.metadata.total();
    let mut values: Vec<(&str, f64)> = vec![
        ("n_units", n_countries as f64),
        ("final_round", result.final_round as f64),
        (
            "cold_war_flag",
            if result.cold_war_flag { 1.0 } else { 0.0 },
        ),
        ("llm_calls", calls as f64),
        ("llm_cache_hits", result.metadata.cache_hits() as f64),
    ];
    if let Some(round) = result.escalation_round {
        values.push(("escalation_round", round as f64));
    }
    // 呼び出しが 1 本も無いときの cache-hit 率は «0» ではなく «定義できない»．
    // 欠測を 0 で埋めず，率の行そのものを書かない．
    if calls > 0 {
        values.push(("llm_cache_hit_rate", result.metadata.cache_hit_rate()));
    }
    run.log_metrics(SCOPE, &values)
        .expect("run スコープの指標の記録に失敗");
}

/// 行動主体の id．国の `AgentId` をそのまま使う．
fn unit_id(actor: u64) -> String {
    format!("country-{actor}")
}

// --------------------------------------------------------------------------- //
// 論文が報告した値
// --------------------------------------------------------------------------- //

/// 論文 Table 2 が報告した WWI の精度 (GPT-4 バックボーン; round 6・7 回平均)．
///
/// 入るのは **論文が印字した数だけ**である．設計書 §5 が «再現目標» として置いた
/// 帯 (>60% / >75% / 傾向一致) はこの再現実装が決めたものなので行にしない．
/// 同様に `reproduce` のアンカー (`outbreak=1` / `>=2` / `cold_war=1` /
/// `archduke >= null`) は論文の定性的な主張をこの実装が符号化したものであって
/// 論文が印字した数ではないので，`reference.csv` には入れずコンソールに残す．
///
/// 名前は観測側のステップ指標と揃えてある (差分がそのまま取れる)．論文値は
/// «WWI 8 カ国・史実トリガー・round 6» の条件のものなので，どの条件の数かが
/// 後から分かるように `source` に条件を書く．
const PAPER_VALUES: [(&str, f64, &str, &str); 3] = [
    (
        "alliance_mi",
        0.7778,
        "table2",
        "Hua et al. (2024), Table 2 — WWI with a GPT-4 backbone, evaluated at round 6 and \
         averaged over 7 runs: alliance accuracy 77.78%",
    ),
    (
        "declaration_jaccard",
        0.546,
        "table2",
        "Hua et al. (2024), Table 2 — WWI with a GPT-4 backbone, evaluated at round 6 and \
         averaged over 7 runs: war-declaration accuracy 54.60%",
    ),
    (
        "mobilization_jaccard",
        0.9209,
        "table2",
        "Hua et al. (2024), Table 2 — WWI with a GPT-4 backbone, evaluated at round 6 and \
         averaged over 7 runs: mobilization accuracy 92.09%",
    ),
];

/// 論文 Table 2 の報告値を出典付きで `reference.csv` に書く．
///
/// 呼ぶのは `run` サブコマンドだけである — この 3 指標を «その条件で» 直接測る
/// のは 1 本のシミュレーションであり，`run` の既定 (`--scenario wwi --trigger
/// archduke-assassination --rounds 6`) が論文 Table 2 の条件そのものだからである．
/// 掃引の子は条件を振るために回すもので，`reproduce` の親は条件をまたいだ集約を
/// 持つ別の粒度なので，どちらにも同じ論文値は置かない．
pub fn log_paper_references(run: &mut Run) {
    for (name, value, target, source) in PAPER_VALUES {
        run.log_reference(name, value)
            .scope(SCOPE)
            .target(target)
            .source(source)
            .send()
            .unwrap_or_else(|e| panic!("{name} の論文値の記録に失敗: {e}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runvault::meta::TargetKind;

    #[test]
    fn the_work_id_agrees_with_the_arxiv_id() {
        let research: runvault::meta::Research = replication().into();
        let work = research.work.expect("再現実験なので work がある");
        assert_eq!(work.work_id, "arxiv:2311.17227");
        assert_eq!(work.paper_id.as_deref(), Some("P00001798"));
    }

    #[test]
    fn the_targets_are_the_table_and_the_two_headline_claims() {
        let research: runvault::meta::Research = replication().into();
        assert_eq!(research.targets.len(), 3);
        assert!(matches!(research.targets[0].kind, TargetKind::Table));
        assert!(matches!(research.targets[1].kind, TargetKind::Claim));
        assert!(matches!(research.targets[2].kind, TargetKind::Claim));
    }

    #[test]
    fn a_run_that_reproduces_the_paper_passes_the_research_checks() {
        let research: runvault::meta::Research = replication().into();
        runvault::verify::check_research(&research).expect("research の検査に失敗");
    }

    /// `reference.csv` の `target_id` は `research.targets[]` に無いと
    /// `verify` が撥ねる．
    #[test]
    fn every_paper_value_points_at_a_declared_target() {
        let research: runvault::meta::Research = replication().into();
        for (name, _, target, _) in PAPER_VALUES {
            assert!(
                research.targets.iter().any(|t| t.target_id == target),
                "{name} の target `{target}` が research.targets[] にありません"
            );
        }
    }

    #[test]
    fn the_provider_comes_from_the_endpoint() {
        assert_eq!(llm_block("m", "mock://scripted", 0.0).provider, "mock");
        assert_eq!(
            llm_block("m", "https://api.openai.com/v1", 0.0).provider,
            "openai"
        );
        assert_eq!(
            llm_block("m", "http://localhost:11434", 0.0).provider,
            "ollama"
        );
    }
}
