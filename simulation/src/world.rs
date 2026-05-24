//! socsim フレームワーク上の WarAgent 世界状態．
//!
//! エージェント = 移動する空間主体ではなく，外交ラウンドを通じて相互作用する
//! 少数の «国» である (WWI=8)．したがって `socsim-grid` は採用せず，各国の状態を
//! `BTreeMap<AgentId, Country>` に，自国視点の関係 (Board) を
//! `BTreeMap<AgentId, Board>` に保持する．Board の `relations` (相手国 → W/M/T/P)
//! が国際関係の **source of truth** である．これに加えて，同盟クラスタ抽出・
//! 可視化のためのグローバル同盟グラフを `socsim-net::SocialNetwork` として
//! 任意に保持する (metrics 用; Board から導出されるビュー)．
//!
//! `#[derive(Clone, Serialize, Deserialize)]` でスナップショット (save/resume) と
//! 感度分析の比較に対応する．`agent_ids()` は `countries` の昇順キーを返し
//! 決定論を担保する (socsim コア層)．

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};

use crate::board::{Board, Relation, Stick};

/// 7 行動空間 (論文の action space)．
///
/// 各国 LLM が 1 ラウンドに 1 つ選ぶ．`DeclareWar`/`Alliance`/`NonAggression`/
/// `Peace` は対象国を伴う対外行動，`Mobilize`/`Wait` は対内/不作為，`Message` は
/// 通信のみ (関係を直接変えない)．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionKind {
    /// 何もしない (現状維持)．
    Wait,
    /// 総動員する (Stick: MO を立てる)．
    Mobilize,
    /// 対象国へ宣戦布告する (Board: W)．
    DeclareWar,
    /// 対象国へ軍事同盟を要請する (受諾で Board: M)．
    Alliance,
    /// 対象国へ不干渉条約を提案する (受諾で Board: T)．
    NonAggression,
    /// 対象国へ和平を提案する (Board: P へ戻す)．
    Peace,
    /// 対象国へメッセージを送る (関係は変えない; 記憶/伝播のみ)．
    Message,
}

impl ActionKind {
    /// CSV/ログ用の小文字ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            ActionKind::Wait => "wait",
            ActionKind::Mobilize => "mobilize",
            ActionKind::DeclareWar => "declare_war",
            ActionKind::Alliance => "alliance",
            ActionKind::NonAggression => "non_aggression",
            ActionKind::Peace => "peace",
            ActionKind::Message => "message",
        }
    }

    /// 対象国を要する対外行動か (Wait/Mobilize は不要)．
    pub fn needs_target(&self) -> bool {
        matches!(
            self,
            ActionKind::DeclareWar
                | ActionKind::Alliance
                | ActionKind::NonAggression
                | ActionKind::Peace
                | ActionKind::Message
        )
    }
}

/// 行動の公開性 (publicity)．伝播範囲を決める．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Publicity {
    /// 公開: 全エージェントへブロードキャストされる．
    Public,
    /// 秘密: 対象国にのみ伝わる．
    Secret,
}

impl Publicity {
    pub fn label(&self) -> &'static str {
        match self {
            Publicity::Public => "public",
            Publicity::Secret => "secret",
        }
    }
}

/// 1 ラウンドで 1 国が選んだ行動 (Decision で蓄積 → Interaction で解決)．
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    /// 行動主体の `AgentId`．
    pub actor: AgentId,
    /// 行動種別．
    pub kind: ActionKind,
    /// 対象国 (対外行動のみ; Wait/Mobilize は None)．
    pub target: Option<AgentId>,
    /// 公開性．
    pub publicity: Publicity,
    /// 何ラウンド目の行動か (0 始まり)．
    pub round: u64,
}

/// 観測・出力用のイベントログ項目．
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    /// ラウンド (0 始まり)．
    pub round: u64,
    /// 行動主体の生 `u64`．
    pub actor: u64,
    /// 行動ラベル．
    pub action: String,
    /// 対象国の生 `u64` (なければ空)．
    pub target: Option<u64>,
    /// 公開性ラベル．
    pub publicity: String,
}

/// 国の攻撃性スタンス．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stance {
    /// 保守的 (防御重視)．
    Conservative,
    /// 中立．
    Neutral,
    /// 攻撃的 (拡張・先制重視)．
    Aggressive,
}

impl Stance {
    pub fn label(&self) -> &'static str {
        match self {
            Stance::Conservative => "conservative",
            Stance::Neutral => "neutral",
            Stance::Aggressive => "aggressive",
        }
    }

    /// プロンプト注入用の一文．
    pub fn describe(&self) -> &'static str {
        match self {
            Stance::Conservative => {
                "Your government is conservative and risk-averse; you prefer defensive treaties \
                 and avoid initiating conflict."
            }
            Stance::Neutral => "Your government weighs costs and benefits pragmatically.",
            Stance::Aggressive => {
                "Your government is aggressive and expansionist; you are willing to mobilize and \
                 strike pre-emptively to secure your interests."
            }
        }
    }
}

/// 1 国の 6 次元プロフィール (論文の country profile)．
///
/// 自由記述テキストの 6 次元 (Leadership / Military Capability / Resources /
/// Historical Background / Key Policy / Public Morale) を保持し，意思決定
/// プロンプトに注入する．
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Profile {
    /// 指導部の性格・体制．
    pub leadership: String,
    /// 軍事力．
    pub military: String,
    /// 資源・経済力．
    pub resources: String,
    /// 歴史的背景 (係争・恨み等; 開戦の触媒になりうる)．
    pub history: String,
    /// 主要政策 (孤立主義/介入主義など)．
    pub policy: String,
    /// 国民士気．
    pub morale: String,
}

/// 1 国エージェントの状態 (6 次元プロフィール + 動的状態)．
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Country {
    /// 匿名化国名 (例 "Country B")．LLM が史実を «記憶» で再生するのを防ぐ．
    pub name: String,
    /// 6 次元プロフィール．
    pub profile: Profile,
    /// 攻撃性スタンス．
    pub stance: Stance,
    /// Stick (対内状態; 総動員 MO)．
    pub stick: Stick,
    /// 受信済み (自分宛 or 公開) 行動の記憶 (直近のみ保持)．
    pub memory: Vec<String>,
}

impl Country {
    /// プロフィール + スタンスから国を作る．
    pub fn new(name: impl Into<String>, profile: Profile, stance: Stance) -> Self {
        Country {
            name: name.into(),
            profile,
            stance,
            stick: Stick::default(),
            memory: Vec::new(),
        }
    }

    /// 総動員したか (Stick: MO)．
    pub fn mobilized(&self) -> bool {
        self.stick.mobilized
    }
}

/// WarAgent シミュレーションの世界状態．
#[derive(Clone, Serialize, Deserialize)]
pub struct WarWorld {
    /// シミュレーションクロック (1 tick = 1 ラウンド)．
    pub clock: SimClock,
    /// 国エージェント (ソート済みキー)．
    pub countries: BTreeMap<AgentId, Country>,
    /// 各国固有の Board (相手国 → W/M/T/P; **source of truth**)．
    pub boards: BTreeMap<AgentId, Board>,
    /// 今ラウンドで蓄積された行動 (Decision → Interaction)．
    pub pending_actions: Vec<Action>,
    /// 配送待ち行動 (publicity に応じて伝播; 各国の inbox)．
    pub inbox: BTreeMap<AgentId, Vec<Action>>,
    /// 全イベントログ (観測・出力用)．
    pub event_log: Vec<Event>,
    /// ラウンド番号 (0 始まり)．
    pub round: u64,
    /// 第 1 ラウンドに注入するトリガー説明 (None なら null トリガー)．
    pub trigger: Option<String>,
    /// トリガーを既に注入したか．
    pub trigger_injected: bool,
}

impl WarWorld {
    /// 国数 N．
    pub fn n_countries(&self) -> usize {
        self.countries.len()
    }

    /// 現在のラウンド (0 始まり)．
    ///
    /// socsim エンジンはステップ先頭で `tick()` するため，クロックは 1..=rounds を
    /// 走る．本モデルはラウンドを 0 始まり (0..rounds) で扱うので `t() - 1` を返す．
    pub fn current_round(&self) -> u64 {
        self.clock.t().saturating_sub(1)
    }

    /// `AgentId` → 匿名化国名 の写像 (Translate 用)．
    pub fn name_map(&self) -> BTreeMap<AgentId, String> {
        self.countries
            .iter()
            .map(|(id, c)| (*id, c.name.clone()))
            .collect()
    }

    /// 自国視点で，指定の関係種別を持つ相手国を列挙する (Board ベース)．
    pub fn counterparts(&self, owner: AgentId, rel: Relation) -> Vec<AgentId> {
        self.boards
            .get(&owner)
            .map(|b| b.counterparts(rel))
            .unwrap_or_default()
    }

    /// 終了時点の宣戦布告 (W) 関係対の総数 (順序対の半分; 無向で数える)．
    ///
    /// `(a,b)` と `(b,a)` の双方が W のとき 1 対と数える．片側だけ W のときも
    /// 紛争対として 1 と数える (宣戦は片務的に成立しうるため)．
    pub fn n_war_pairs(&self) -> usize {
        use std::collections::BTreeSet;
        let mut pairs: BTreeSet<(u64, u64)> = BTreeSet::new();
        for (owner, board) in &self.boards {
            for other in board.counterparts(Relation::War) {
                let (a, b) = if owner.0 <= other.0 {
                    (owner.0, other.0)
                } else {
                    (other.0, owner.0)
                };
                pairs.insert((a, b));
            }
        }
        pairs.len()
    }
}

impl WorldState for WarWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        self.countries.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with_wars() -> WarWorld {
        let mut countries = BTreeMap::new();
        let mut boards = BTreeMap::new();
        for i in 0..3u64 {
            countries.insert(
                AgentId(i),
                Country::new(format!("Country {i}"), Profile::default(), Stance::Neutral),
            );
            boards.insert(AgentId(i), Board::new(AgentId(i)));
        }
        // 0<->1 mutual war, 1->2 one-sided war.
        boards
            .get_mut(&AgentId(0))
            .unwrap()
            .set_relation(AgentId(1), Relation::War);
        boards
            .get_mut(&AgentId(1))
            .unwrap()
            .set_relation(AgentId(0), Relation::War);
        boards
            .get_mut(&AgentId(1))
            .unwrap()
            .set_relation(AgentId(2), Relation::War);
        WarWorld {
            clock: SimClock::new(6),
            countries,
            boards,
            pending_actions: Vec::new(),
            inbox: BTreeMap::new(),
            event_log: Vec::new(),
            round: 0,
            trigger: None,
            trigger_injected: false,
        }
    }

    #[test]
    fn agent_ids_sorted() {
        let w = world_with_wars();
        assert_eq!(w.agent_ids(), vec![AgentId(0), AgentId(1), AgentId(2)]);
    }

    #[test]
    fn counts_war_pairs_undirected() {
        let w = world_with_wars();
        // {0,1} and {1,2} → 2 pairs.
        assert_eq!(w.n_war_pairs(), 2);
    }

    #[test]
    fn action_kind_target_requirement() {
        assert!(!ActionKind::Wait.needs_target());
        assert!(!ActionKind::Mobilize.needs_target());
        assert!(ActionKind::DeclareWar.needs_target());
        assert!(ActionKind::Alliance.needs_target());
    }
}
