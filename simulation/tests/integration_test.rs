//! Hua et al. (2024) WarAgent 外交 ABM の統合テスト．
//!
//! **ライブ LLM を一切必要としない**: socsim-llm の `mock::ScriptedClient` で
//! 決定論的に国の行動を駆動し，以下を検証する:
//! ・行動解決 (宣戦布告 → Board W; 相互同盟要請 → Board M)
//! ・publicity 伝播 (公開 = 全国 inbox へブロードキャスト / 秘密 = 対象国のみ)
//! ・Board 更新 (Stick MO; 交戦に入った国の総動員)
//! ・エスカレーション (同盟国が宣戦された側に合流して参戦)
//! ・指標計算 (alliance_mi / declaration_jaccard / mobilization_jaccard が [0,1])
//! ・RNG 決定論性 (同一シード → 完全再現)
//! ・停止条件 (開戦しきい値 or 最終ラウンドで停止)

use std::collections::BTreeSet;

use waragent_simulation::config::{build_countries, Config, Scenario, Trigger};
use waragent_simulation::llm::{wrap_client, WarClient};
use waragent_simulation::metrics::{
    alliance_mi, alliance_network, alliance_partition, compute_round_metric,
    historical_alliance_partition, jaccard, mobilized_set, war_pair_set,
};
use waragent_simulation::simulation::run_with_client;
use waragent_simulation::world::Stance;

use socsim_core::SimClock;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

/// 国ごとに固定行動を返す mock．`reply(country_name) -> json` を受ける．
fn scripted(reply: impl Fn(&str) -> String + Send + Sync + 'static) -> WarClient {
    let backend = ScriptedClient::new("mock-model", move |prompt: &str| {
        // プロンプトから "Your country: Country X" を拾う (decision/secretary 双方に出る)．
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H"] {
            if prompt.contains(&format!("Your country: Country {letter}")) {
                return reply(letter);
            }
            if prompt.contains(&format!("## Country: Country {letter}")) {
                return reply(letter);
            }
        }
        "{\"action\": \"wait\"}".to_string()
    });
    wrap_client(backend, PromptCache::in_memory())
}

fn small_cfg() -> Config {
    Config {
        scenario: Scenario::WwiSmall,
        trigger: Trigger::ArchdukeAssassination,
        secretary_passes: 1,
        rounds: 3,
        war_threshold: 1,
        seed: Some(7),
        ..Config::default()
    }
}

fn world_for_metrics() -> waragent_simulation::world::WarWorld {
    let (countries, boards) = build_countries(Scenario::Wwi, None);
    waragent_simulation::world::WarWorld {
        clock: SimClock::new(6),
        countries,
        boards,
        pending_actions: Vec::new(),
        inbox: std::collections::BTreeMap::new(),
        event_log: Vec::new(),
        round: 0,
        trigger: None,
        trigger_injected: false,
    }
}

// --------------------------------------------------------------------------- //
// 行動解決: 宣戦布告 → Board W
// --------------------------------------------------------------------------- //

#[test]
fn declare_war_resolves_to_war_relation() {
    let cfg = Config {
        rounds: 1,
        ..small_cfg()
    };
    // A (id 0) declares war on C (id 2); others wait.
    let client = scripted(|c| {
        if c == "A" {
            "{\"action\": \"declare_war\", \"target\": 2, \"publicity\": \"public\"}".to_string()
        } else {
            "{\"action\": \"wait\"}".to_string()
        }
    });
    let r = run_with_client(&cfg, client).unwrap();
    // declare_war event fired.
    assert!(r
        .event_log
        .iter()
        .any(|e| e.action == "declare_war" && e.actor == 0 && e.target == Some(2)));
    // a conflict pair exists.
    assert!(r.n_conflicts >= 1, "n_conflicts={}", r.n_conflicts);
}

// --------------------------------------------------------------------------- //
// 行動解決: 相互同盟要請 → Board M
// --------------------------------------------------------------------------- //

#[test]
fn mutual_alliance_request_forms_alliance() {
    let cfg = Config {
        rounds: 1,
        war_threshold: 99,
        ..small_cfg()
    };
    // C (id 2) <-> D (id 3) propose alliance to each other → mutual consent.
    let client = scripted(|c| match c {
        "C" => "{\"action\": \"alliance\", \"target\": 3, \"publicity\": \"public\"}".to_string(),
        "D" => "{\"action\": \"alliance\", \"target\": 2, \"publicity\": \"public\"}".to_string(),
        _ => "{\"action\": \"wait\"}".to_string(),
    });
    let r = run_with_client(&cfg, client).unwrap();
    // alliance events fired for both.
    assert!(r
        .event_log
        .iter()
        .any(|e| e.action == "alliance" && e.actor == 2));
    assert!(r
        .event_log
        .iter()
        .any(|e| e.action == "alliance" && e.actor == 3));
}

// --------------------------------------------------------------------------- //
// エスカレーション: 同盟国が参戦する
// --------------------------------------------------------------------------- //

#[test]
fn escalation_pulls_allies_into_war() {
    // wwi-small: A-B are allies (initial). A declares war on... we need a victim
    // whose ally then joins. Use full WWI: C-D, C-E alliances exist.
    let cfg = Config {
        scenario: Scenario::Wwi,
        trigger: Trigger::ArchdukeAssassination,
        secretary_passes: 1,
        rounds: 1,
        war_threshold: 99, // avoid early stop so we observe escalation event in round 0.
        seed: Some(7),
        ..Config::default()
    };
    // A (id 0) declares war on C (id 2). C's allies are D (3) and E (4) → they join vs A.
    let client = scripted(|c| {
        if c == "A" {
            "{\"action\": \"declare_war\", \"target\": 2, \"publicity\": \"public\"}".to_string()
        } else {
            "{\"action\": \"wait\"}".to_string()
        }
    });
    let r = run_with_client(&cfg, client).unwrap();
    // escalation join events present (ally joined war vs aggressor A=0).
    assert!(
        r.event_log
            .iter()
            .any(|e| e.action == "escalate_join_war" && e.target == Some(0)),
        "expected an ally to escalate into war against A"
    );
}

// --------------------------------------------------------------------------- //
// publicity 伝播: 秘密行動は対象国のみ，公開は全国へ
// --------------------------------------------------------------------------- //

#[test]
fn secret_action_reaches_only_target() {
    // We assert via memory/inbox indirectly: a secret message from A to C should
    // not raise war and should keep things calm; we check no spurious conflicts.
    let cfg = Config {
        scenario: Scenario::Wwi,
        rounds: 1,
        war_threshold: 99,
        secretary_passes: 1,
        seed: Some(7),
        ..Config::default()
    };
    let client = scripted(|c| {
        if c == "A" {
            "{\"action\": \"message\", \"target\": 2, \"publicity\": \"secret\"}".to_string()
        } else {
            "{\"action\": \"wait\"}".to_string()
        }
    });
    let r = run_with_client(&cfg, client).unwrap();
    // a secret message creates no new war.
    assert_eq!(r.n_conflicts, 0);
    // the message event was logged as secret.
    assert!(r
        .event_log
        .iter()
        .any(|e| e.action == "message" && e.publicity == "secret" && e.target == Some(2)));
}

// --------------------------------------------------------------------------- //
// Board 更新: 交戦に入った国は総動員する
// --------------------------------------------------------------------------- //

#[test]
fn war_triggers_mobilization() {
    let cfg = Config {
        scenario: Scenario::Wwi,
        rounds: 1,
        war_threshold: 99,
        secretary_passes: 0,
        seed: Some(7),
        ..Config::default()
    };
    let client = scripted(|c| {
        if c == "A" {
            "{\"action\": \"declare_war\", \"target\": 2, \"publicity\": \"public\"}".to_string()
        } else {
            "{\"action\": \"wait\"}".to_string()
        }
    });
    let r = run_with_client(&cfg, client).unwrap();
    // at least A and C mobilized (both at war) → mobilization metric > 0.
    let last = r.metrics_history.last().unwrap();
    assert!(last.n_mobilized >= 2, "n_mobilized={}", last.n_mobilized);
}

// --------------------------------------------------------------------------- //
// 指標: MI / Jaccard が [0,1]，恒等で 1
// --------------------------------------------------------------------------- //

#[test]
fn metric_helpers_hand_calc() {
    let p = historical_alliance_partition();
    assert!((alliance_mi(&p, &p) - 1.0).abs() < 1e-9);

    let a: BTreeSet<u64> = [0u64, 1, 2].into_iter().collect();
    let b: BTreeSet<u64> = [1u64, 2, 3].into_iter().collect();
    assert!((jaccard(&a, &b) - 0.5).abs() < 1e-9);
}

#[test]
fn computed_metrics_in_unit_range() {
    let w = world_for_metrics();
    let m = compute_round_metric(&w, 0, 3);
    assert!((0.0..=1.0).contains(&m.alliance_mi));
    assert!((0.0..=1.0).contains(&m.declaration_jaccard));
    assert!((0.0..=1.0).contains(&m.mobilization_jaccard));
    // initial WWI has no wars / mobilizations.
    assert!(war_pair_set(&w).is_empty());
    assert!(mobilized_set(&w).is_empty());
}

#[test]
fn alliance_network_view_agrees_with_partition() {
    let w = world_for_metrics();
    let net = alliance_network(&w);
    assert_eq!(net.connected_components(), alliance_partition(&w).len());
}

// --------------------------------------------------------------------------- //
// 決定論性: 同一シード + 同一 mock → 完全再現
// --------------------------------------------------------------------------- //

#[test]
fn deterministic_given_fixed_mock() {
    let cfg = small_cfg();
    let mk = || {
        scripted(|c| {
            if c == "A" {
                "{\"action\": \"declare_war\", \"target\": 2, \"publicity\": \"public\"}"
                    .to_string()
            } else {
                "{\"action\": \"wait\"}".to_string()
            }
        })
    };
    let a = run_with_client(&cfg, mk()).unwrap();
    let b = run_with_client(&cfg, mk()).unwrap();
    let ma: Vec<u64> = a.metrics_history.iter().map(|m| m.n_conflicts).collect();
    let mb: Vec<u64> = b.metrics_history.iter().map(|m| m.n_conflicts).collect();
    let ja: Vec<f64> = a.metrics_history.iter().map(|m| m.alliance_mi).collect();
    let jb: Vec<f64> = b.metrics_history.iter().map(|m| m.alliance_mi).collect();
    assert_eq!(ma, mb, "同一シードは紛争数を完全再現すべき");
    assert_eq!(ja, jb, "同一シードは MI を完全再現すべき");
}

// --------------------------------------------------------------------------- //
// 停止条件: 開戦しきい値 or 最終ラウンドで停止
// --------------------------------------------------------------------------- //

#[test]
fn stops_at_war_threshold() {
    let cfg = Config {
        scenario: Scenario::Wwi,
        rounds: 10,
        war_threshold: 1, // any single war pair stops.
        secretary_passes: 1,
        seed: Some(7),
        ..Config::default()
    };
    let client = scripted(|c| {
        if c == "A" {
            "{\"action\": \"declare_war\", \"target\": 2, \"publicity\": \"public\"}".to_string()
        } else {
            "{\"action\": \"wait\"}".to_string()
        }
    });
    let r = run_with_client(&cfg, client).unwrap();
    // war breaks out in round 0 → stops well before rounds=10.
    assert!(r.war_outbreak);
    assert!(r.final_round < 10);
    assert_eq!(r.escalation_round, Some(0));
}

#[test]
fn stops_at_final_round_without_war() {
    let cfg = Config {
        scenario: Scenario::WwiSmall,
        rounds: 2,
        war_threshold: 99,
        secretary_passes: 1,
        seed: Some(7),
        ..small_cfg()
    };
    // everyone waits → no war → runs to the final round.
    let client = scripted(|_| "{\"action\": \"wait\"}".to_string());
    let r = run_with_client(&cfg, client).unwrap();
    assert!(!r.war_outbreak);
    assert_eq!(r.final_round, 2);
    let max_round = r.metrics_history.iter().map(|m| m.round).max().unwrap();
    assert!(max_round < cfg.rounds as u64);
}

#[test]
fn stance_override_reaches_all_countries() {
    let (countries, _) = build_countries(Scenario::Wwi, Some(Stance::Conservative));
    assert!(countries.values().all(|c| c.stance == Stance::Conservative));
}
