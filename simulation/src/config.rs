//! シミュレーション設定 + シナリオ定義．
//!
//! Hua et al. (2024) WarAgent のコアモデル (LLM 駆動の外交ラウンド ABM) と感度分析
//! パラメータを保持する [`Config`]，および匿名化された WWI 8 カ国シナリオの
//! 初期化データを定義する．Scenario 列挙・secretary_passes を拡張点として残す
//! (WWII / Warring-States は Phase 3 で追加)．

use std::collections::BTreeMap;

use serde::Serialize;
use socsim_core::AgentId;

use crate::board::{Board, Relation};
use crate::world::{Country, Profile, Stance};

// --------------------------------------------------------------------------- //
// Scenario
// --------------------------------------------------------------------------- //

/// 歴史シナリオ (拡張点)．Phase 1 は WWI のみ実装する．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// 第一次世界大戦 (8 カ国; Phase 1)．
    Wwi,
    /// 縮小 WWI (4 カ国; 軽量スモーク用)．
    WwiSmall,
    // 拡張点: Wwii, WarringStates (Phase 3)．
}

impl Scenario {
    pub fn label(&self) -> &'static str {
        match self {
            Scenario::Wwi => "wwi",
            Scenario::WwiSmall => "wwi-small",
        }
    }
}

/// 文字列から [`Scenario`] をパースする．
pub fn parse_scenario(s: &str) -> Result<Scenario, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "wwi" | "ww1" | "wwone" => Ok(Scenario::Wwi),
        "wwi-small" | "wwi_small" | "small" => Ok(Scenario::WwiSmall),
        _ => Err(format!("不正なシナリオ: \"{s}\" (wwi / wwi-small)")),
    }
}

// --------------------------------------------------------------------------- //
// Trigger
// --------------------------------------------------------------------------- //

/// トリガーイベントの強度 (感度分析の主軸)．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// null トリガー (注入なし; 冷戦の検証)．
    Null,
    /// 海上事件 (低強度)．
    NavalIncident,
    /// ダーダネルス海峡封鎖 (高強度)．
    Dardanelles,
    /// 大公暗殺 (史実トリガー; 最高強度)．
    ArchdukeAssassination,
}

impl Trigger {
    pub fn label(&self) -> &'static str {
        match self {
            Trigger::Null => "null",
            Trigger::NavalIncident => "naval-incident",
            Trigger::Dardanelles => "dardanelles",
            Trigger::ArchdukeAssassination => "archduke-assassination",
        }
    }

    /// 第 1 ラウンドに注入する説明文 (Null は None)．国名は匿名化済みを使う．
    pub fn description(&self) -> Option<&'static str> {
        match self {
            Trigger::Null => None,
            Trigger::NavalIncident => Some(
                "A naval skirmish has occurred between two great powers in contested waters. \
                 Casualties are limited but the press is inflamed.",
            ),
            Trigger::Dardanelles => Some(
                "A major strait controlling access to a great power's heartland has been blockaded, \
                 cutting off vital trade. The affected power demands immediate reversal.",
            ),
            Trigger::ArchdukeAssassination => Some(
                "The heir to the throne of one of the great powers has been assassinated by a \
                 nationalist from a neighbouring state. An ultimatum is being drafted.",
            ),
        }
    }
}

/// 文字列から [`Trigger`] をパースする．
pub fn parse_trigger(s: &str) -> Result<Trigger, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "null" | "none" => Ok(Trigger::Null),
        "naval-incident" | "naval" | "naval_incident" => Ok(Trigger::NavalIncident),
        "dardanelles" => Ok(Trigger::Dardanelles),
        "archduke-assassination" | "archduke" | "assassination" => {
            Ok(Trigger::ArchdukeAssassination)
        }
        _ => Err(format!(
            "不正なトリガー: \"{s}\" (null / naval-incident / dardanelles / archduke-assassination)"
        )),
    }
}

/// 文字列から [`Stance`] をパースする．
pub fn parse_stance(s: &str) -> Result<Stance, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "conservative" | "defensive" => Ok(Stance::Conservative),
        "neutral" => Ok(Stance::Neutral),
        "aggressive" | "offensive" => Ok(Stance::Aggressive),
        _ => Err(format!(
            "不正なスタンス: \"{s}\" (conservative / neutral / aggressive)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// 定義は `socsim-llm` に集約済み (各 replication で同一だった struct を統合)．
/// `crate::config::LlmSettings` パスは re-export で温存する．
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
#[derive(Debug, Clone)]
pub struct Config {
    /// シナリオ (WWI 等)．
    pub scenario: Scenario,
    /// トリガーイベント．
    pub trigger: Trigger,
    /// 全国に上書きするスタンス (None ならシナリオ既定)．
    pub stance_override: Option<Stance>,
    /// 秘書検証パス数 (各国の最終行動を検証する回数; 既定 1，LLM 呼び出しを有界化)．
    pub secretary_passes: usize,
    /// 最大ラウンド数 (既定 6)．
    pub rounds: usize,
    /// 世界大戦勃発の宣戦布告対しきい値 (W 対の数がこれ以上で勃発)．
    pub war_threshold: usize,

    /// 乱数シード (None ならランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
    /// 結果出力ディレクトリ．
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            scenario: Scenario::Wwi,
            trigger: Trigger::ArchdukeAssassination,
            stance_override: None,
            secretary_passes: 1,
            rounds: 6,
            war_threshold: 3,
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

/// `run` の試行シードを派生する (試行 index で独立化)．
pub fn derive_run_seed(base: u64, run_idx: usize) -> u64 {
    socsim_core::derive_seed(base, &[run_idx as u64])
}

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub scenario: String,
    pub trigger: String,
    pub stance_override: Option<String>,
    pub secretary_passes: usize,
    pub rounds: usize,
    pub war_threshold: usize,
    pub n_countries: usize,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            scenario: self.scenario.label().to_string(),
            trigger: self.trigger.label().to_string(),
            stance_override: self.stance_override.map(|s| s.label().to_string()),
            secretary_passes: self.secretary_passes,
            rounds: self.rounds,
            war_threshold: self.war_threshold,
            n_countries: scenario_country_count(self.scenario),
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
}

// --------------------------------------------------------------------------- //
// シナリオ初期化データ (WWI 8 カ国; 匿名化)
// --------------------------------------------------------------------------- //

/// シナリオの国数を返す．
pub fn scenario_country_count(scenario: Scenario) -> usize {
    match scenario {
        Scenario::Wwi => 8,
        Scenario::WwiSmall => 4,
    }
}

/// シナリオの初期化済み «国 + 自国 Board» を返す (匿名化国名)．
///
/// WWI の 8 カ国を匿名キー (Country A..H) で表現する．史実の同盟構造 (三国協商 vs
/// 三国同盟系) を初期 Board の M (alliance) / T (non-aggression) で与え，史実恨み
/// (仏独間アルザス・ロレーヌ) を T 不在の緊張として残す．国名・プロフィールは
/// 匿名化し，LLM が史実を «記憶» で再生するのを防ぐ (論文 §脱匿名化)．
///
/// インデックス↔史実の対応 (脱匿名化; Phase 3 の検証用にコメントとして残す):
/// A=Germany, B=Austria-Hungary, C=France, D=United Kingdom, E=Russia,
/// F=Italy, G=Ottoman Empire, H=Serbia.
pub fn build_countries(
    scenario: Scenario,
    stance_override: Option<Stance>,
) -> (BTreeMap<AgentId, Country>, BTreeMap<AgentId, Board>) {
    let specs = match scenario {
        Scenario::Wwi => wwi_specs(),
        Scenario::WwiSmall => wwi_specs().into_iter().take(4).collect(),
    };
    let n = specs.len();

    let mut countries: BTreeMap<AgentId, Country> = BTreeMap::new();
    let mut boards: BTreeMap<AgentId, Board> = BTreeMap::new();

    for (i, spec) in specs.iter().enumerate() {
        let id = AgentId(i as u64);
        let stance = stance_override.unwrap_or(spec.stance);
        countries.insert(id, Country::new(spec.name, spec.profile(), stance));
        boards.insert(id, Board::new(id));
    }

    // 初期同盟/不干渉を Board に書く (対称に張る; 範囲内のペアのみ)．
    for &(a, b, rel) in initial_relations(scenario).iter() {
        if (a as usize) < n && (b as usize) < n {
            boards
                .get_mut(&AgentId(a))
                .unwrap()
                .set_relation(AgentId(b), rel);
            boards
                .get_mut(&AgentId(b))
                .unwrap()
                .set_relation(AgentId(a), rel);
        }
    }

    (countries, boards)
}

/// 1 国の匿名化スペック．
struct CountrySpec {
    name: &'static str,
    stance: Stance,
    leadership: &'static str,
    military: &'static str,
    resources: &'static str,
    history: &'static str,
    policy: &'static str,
    morale: &'static str,
}

impl CountrySpec {
    fn profile(&self) -> Profile {
        Profile {
            leadership: self.leadership.to_string(),
            military: self.military.to_string(),
            resources: self.resources.to_string(),
            history: self.history.to_string(),
            policy: self.policy.to_string(),
            morale: self.morale.to_string(),
        }
    }
}

/// WWI 8 カ国の匿名化スペック (A..H)．
fn wwi_specs() -> Vec<CountrySpec> {
    vec![
        CountrySpec {
            name: "Country A",
            stance: Stance::Aggressive,
            leadership: "A centralized monarchy with an assertive general staff.",
            military:
                "The strongest standing army on the continent, highly mechanized for the era.",
            resources: "An advanced industrial base but limited overseas supply lines.",
            history: "A long rivalry and a disputed border province with Country C.",
            policy: "Seeks to secure its position before rivals encircle it.",
            morale: "High; the public is confident and nationalistic.",
        },
        CountrySpec {
            name: "Country B",
            stance: Stance::Aggressive,
            leadership: "A multi-ethnic empire ruled by an aging court.",
            military: "Large but unevenly equipped; internal nationalities strain cohesion.",
            resources: "Moderate; agrarian heartland with some industry.",
            history: "Bitter friction with neighbouring Country H over border nationalism.",
            policy: "Wants to punish Country H and preserve imperial prestige.",
            morale: "Mixed across its many nationalities.",
        },
        CountrySpec {
            name: "Country C",
            stance: Stance::Conservative,
            leadership: "A republic with a cautious but proud government.",
            military: "A strong army oriented toward defending its eastern border.",
            resources: "Solid industry and a global colonial supply network.",
            history: "Lost a border province to Country A in a past war; deep resentment remains.",
            policy: "Defensive, but will honour its alliance commitments.",
            morale: "Determined, with a desire to recover lost territory.",
        },
        CountrySpec {
            name: "Country D",
            stance: Stance::Conservative,
            leadership: "A constitutional government with a powerful navy ministry.",
            military: "The dominant naval power; a small but professional army.",
            resources: "Vast colonial and maritime resources.",
            history: "Wary of any single power dominating the continent.",
            policy:
                "Balance-of-power; intervenes if a rival grows too strong or a neutral is invaded.",
            morale: "Steady and confident in naval supremacy.",
        },
        CountrySpec {
            name: "Country E",
            stance: Stance::Neutral,
            leadership: "A vast autocratic empire with a slow-moving bureaucracy.",
            military: "Enormous manpower but poor logistics and equipment.",
            resources: "Abundant raw materials, weak industrialization.",
            history: "Sees itself as protector of smaller kindred states such as Country H.",
            policy: "Will mobilize to defend client states it patronises.",
            morale: "Fragile; domestic unrest simmers beneath patriotism.",
        },
        CountrySpec {
            name: "Country F",
            stance: Stance::Neutral,
            leadership: "A young kingdom balancing rival blocs opportunistically.",
            military: "Modest; modernizing but not first-rate.",
            resources: "Limited industry; dependent on imports.",
            history: "Nominal ties to Country A and B but unresolved claims against Country B.",
            policy: "Joins whichever side best serves its territorial ambitions.",
            morale: "Eager for gains but cautious of overreach.",
        },
        CountrySpec {
            name: "Country G",
            stance: Stance::Neutral,
            leadership: "A reforming empire seeking to halt its decline.",
            military: "Weakened but controls strategic straits.",
            resources: "Strained; geography gives it leverage over trade routes.",
            history: "Long enmity with Country E over territory and straits.",
            policy: "Aligns with whoever can guarantee its integrity against Country E.",
            morale: "Anxious but proud.",
        },
        CountrySpec {
            name: "Country H",
            stance: Stance::Aggressive,
            leadership: "A small assertive kingdom riding a wave of nationalism.",
            military: "Small but battle-hardened.",
            resources: "Scarce; relies on a great-power patron (Country E).",
            history: "Aspires to unite kindred peoples currently under Country B's rule.",
            policy: "Defiant toward Country B; counts on Country E's protection.",
            morale: "Fervent and defiant.",
        },
    ]
}

/// WWI の初期関係 (対称; M=alliance / T=non-aggression)．
///
/// 史実の二大陣営をおおまかに与える: A-B-G が一方の同盟系，C-D-E が協商系，
/// E-H は保護関係 (alliance)，F は中立 (どちらにも初期 alliance なし)．
fn initial_relations(scenario: Scenario) -> Vec<(u64, u64, Relation)> {
    match scenario {
        Scenario::Wwi => vec![
            (0, 1, Relation::Alliance),      // A-B 中央同盟
            (1, 5, Relation::NonAggression), // B-F (係争はあるが当初は不戦)
            (2, 3, Relation::Alliance),      // C-D 協商
            (2, 4, Relation::Alliance),      // C-E 協商
            (3, 4, Relation::NonAggression), // D-E
            (4, 7, Relation::Alliance),      // E-H 保護関係
            (0, 6, Relation::NonAggression), // A-G (後に接近)
        ],
        Scenario::WwiSmall => vec![
            (0, 1, Relation::Alliance),      // A-B
            (2, 3, Relation::NonAggression), // C-D
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wwi_has_eight_countries() {
        let (countries, boards) = build_countries(Scenario::Wwi, None);
        assert_eq!(countries.len(), 8);
        assert_eq!(boards.len(), 8);
        // anonymized names.
        assert_eq!(countries[&AgentId(0)].name, "Country A");
        assert_eq!(countries[&AgentId(7)].name, "Country H");
    }

    #[test]
    fn small_scenario_has_four() {
        let (countries, _) = build_countries(Scenario::WwiSmall, None);
        assert_eq!(countries.len(), 4);
    }

    #[test]
    fn initial_alliances_are_symmetric() {
        let (_, boards) = build_countries(Scenario::Wwi, None);
        assert_eq!(
            boards[&AgentId(0)].relation_to(AgentId(1)),
            Relation::Alliance
        );
        assert_eq!(
            boards[&AgentId(1)].relation_to(AgentId(0)),
            Relation::Alliance
        );
    }

    #[test]
    fn stance_override_applies() {
        let (countries, _) = build_countries(Scenario::Wwi, Some(Stance::Aggressive));
        assert!(countries.values().all(|c| c.stance == Stance::Aggressive));
    }

    #[test]
    fn parse_helpers() {
        assert_eq!(parse_scenario("wwi").unwrap(), Scenario::Wwi);
        assert_eq!(parse_trigger("null").unwrap(), Trigger::Null);
        assert_eq!(parse_stance("aggressive").unwrap(), Stance::Aggressive);
        assert!(parse_trigger("bogus").is_err());
    }
}
