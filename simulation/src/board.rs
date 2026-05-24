//! Board (international relations) と Stick (domestic state) の構造体 +
//! ルールベース変換 (Translate)．
//!
//! WarAgent の «Board and Stick context management» (論文 §核心 3) を実装する．
//! 各国は自国視点の Board (対外関係) を固有に保持する «部分情報の原則» に従い，
//! Board は «相手国 → 関係種別 (W/M/T/P)» の写像で表現する．Stick は対内状態
//! (本実装では総動員 MO に絞る)．Translate は Board/Stick をルールベースで
//! パラグラフ化し，マルチターン会話を実質 1 ターンへ «圧縮» してプロンプトに渡す
//! (LLM 呼び出し回数を有界化する設計の核心)．

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::AgentId;

/// 2 国間の関係種別 (論文の Board ラベル W/M/T/P)．
///
/// 各国の Board は «自国 → 相手国» の関係をこの列挙で持つ．`Peace` は «特段の
/// 関係なし (平時)» の既定状態である．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Relation {
    /// W: 宣戦布告 (war declaration; 交戦状態)．
    War,
    /// M: 軍事同盟 (military alliance)．
    Alliance,
    /// T: 不干渉条約 (non-aggression treaty)．
    NonAggression,
    /// P: 和平 / 平時 (peace; 既定)．
    Peace,
}

impl Relation {
    /// Board ラベル (W/M/T/P)．
    pub fn label(&self) -> &'static str {
        match self {
            Relation::War => "W",
            Relation::Alliance => "M",
            Relation::NonAggression => "T",
            Relation::Peace => "P",
        }
    }

    /// 人間可読な関係名 (Translate 用)．
    pub fn describe(&self) -> &'static str {
        match self {
            Relation::War => "at war with",
            Relation::Alliance => "in a military alliance with",
            Relation::NonAggression => "bound by a non-aggression treaty with",
            Relation::Peace => "at peace with",
        }
    }
}

/// 自国視点の Board: «相手国 (AgentId) → 関係種別»．
///
/// 部分情報の原則により各国が固有に保持する．キーは «(自国, 相手国)» の対だが，
/// 1 つの Board は 1 国に属するため «相手国 → 関係» の写像で十分である．設計書の
/// `BTreeMap<(AgentId, AgentId), Relation>` 表記に合わせ，[`WarWorld`] 側では
/// «(owner, other)» の対をキーにした明示的行列としても観測できる
/// ([`Board::pairs`])．
///
/// [`WarWorld`]: crate::world::WarWorld
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Board {
    /// この Board を保持する国の `AgentId`．
    pub owner: AgentId,
    /// 相手国 → 関係種別 (未登録は `Peace` 扱い)．
    pub relations: BTreeMap<AgentId, Relation>,
}

impl Board {
    /// 空の Board を作る (所有国を指定; すべて Peace)．
    pub fn new(owner: AgentId) -> Self {
        Board {
            owner,
            relations: BTreeMap::new(),
        }
    }

    /// 相手国との関係を引く (未登録は `Peace`)．
    pub fn relation_to(&self, other: AgentId) -> Relation {
        self.relations
            .get(&other)
            .copied()
            .unwrap_or(Relation::Peace)
    }

    /// 相手国との関係を設定する (Peace は «関係解消» として削除する)．
    pub fn set_relation(&mut self, other: AgentId, rel: Relation) {
        if rel == Relation::Peace {
            self.relations.remove(&other);
        } else {
            self.relations.insert(other, rel);
        }
    }

    /// «(owner, other)» 対をキーにした明示的行列の行を列挙する (観測・出力用)．
    pub fn pairs(&self) -> impl Iterator<Item = ((AgentId, AgentId), Relation)> + '_ {
        self.relations
            .iter()
            .map(move |(other, rel)| ((self.owner, *other), *rel))
    }

    /// 指定の関係種別を持つ相手国を列挙する (ソート順)．
    pub fn counterparts(&self, rel: Relation) -> Vec<AgentId> {
        self.relations
            .iter()
            .filter(|(_, r)| **r == rel)
            .map(|(id, _)| *id)
            .collect()
    }
}

/// Stick (対内状態)．本実装では総動員フラグ (MO) に絞る．
///
/// 論文の Stick は MO (総動員) / IN (国内不安) / WR (戦時) を持つが，再現コアでは
/// 開戦・エスカレーション動学に直結する MO のみをモデル化する (拡張点)．
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Stick {
    /// MO: 総動員したか．
    pub mobilized: bool,
}

impl Stick {
    /// Stick の総動員ラベル (Translate 用)．
    pub fn label(&self) -> &'static str {
        if self.mobilized {
            "MO (fully mobilized)"
        } else {
            "not mobilized"
        }
    }
}

/// Board/Stick をルールベースでパラグラフ化する (Translate)．
///
/// `names` は `AgentId` → 匿名化国名 (例 "Country B") の写像．自国視点の関係一覧と
/// 総動員状態を 1 つの文脈パラグラフに圧縮し，[`SituationMechanism`] が各国の
/// 意思決定プロンプトに注入する．関係が «Peace のみ» でも «no active treaties or
/// conflicts» と明示し，プロンプト全文 (= キャッシュキー) を決定論化する．
///
/// [`SituationMechanism`]: crate::mechanisms::SituationMechanism
pub fn translate(board: &Board, stick: &Stick, names: &BTreeMap<AgentId, String>) -> String {
    let self_name = names
        .get(&board.owner)
        .cloned()
        .unwrap_or_else(|| format!("Country#{}", board.owner.0));

    let mut active: Vec<(AgentId, Relation)> = board
        .relations
        .iter()
        .filter(|(_, r)| **r != Relation::Peace)
        .map(|(id, r)| (*id, *r))
        .collect();
    active.sort_by_key(|(id, _)| id.0);

    let mut s = String::new();
    s.push_str(&format!(
        "You are the leadership of {self_name}. Your domestic mobilization status: {}.\n",
        stick.label()
    ));
    if active.is_empty() {
        s.push_str(
            "You currently have no active treaties or active conflicts with any other country.\n",
        );
    } else {
        s.push_str("Your current international relations:\n");
        for (other, rel) in active {
            let other_name = names
                .get(&other)
                .cloned()
                .unwrap_or_else(|| format!("Country#{}", other.0));
            s.push_str(&format!(
                "- You are {} {} [{}].\n",
                rel.describe(),
                other_name,
                rel.label()
            ));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> BTreeMap<AgentId, String> {
        let mut m = BTreeMap::new();
        m.insert(AgentId(0), "Country A".to_string());
        m.insert(AgentId(1), "Country B".to_string());
        m.insert(AgentId(2), "Country C".to_string());
        m
    }

    #[test]
    fn peace_is_default_and_removes() {
        let mut b = Board::new(AgentId(0));
        assert_eq!(b.relation_to(AgentId(1)), Relation::Peace);
        b.set_relation(AgentId(1), Relation::Alliance);
        assert_eq!(b.relation_to(AgentId(1)), Relation::Alliance);
        b.set_relation(AgentId(1), Relation::Peace);
        assert_eq!(b.relation_to(AgentId(1)), Relation::Peace);
        assert!(b.relations.is_empty());
    }

    #[test]
    fn counterparts_filters_by_relation() {
        let mut b = Board::new(AgentId(0));
        b.set_relation(AgentId(1), Relation::War);
        b.set_relation(AgentId(2), Relation::Alliance);
        assert_eq!(b.counterparts(Relation::War), vec![AgentId(1)]);
        assert_eq!(b.counterparts(Relation::Alliance), vec![AgentId(2)]);
    }

    #[test]
    fn translate_lists_active_relations() {
        let mut b = Board::new(AgentId(0));
        b.set_relation(AgentId(1), Relation::Alliance);
        b.set_relation(AgentId(2), Relation::War);
        let stick = Stick { mobilized: true };
        let para = translate(&b, &stick, &names());
        assert!(para.contains("Country A"));
        assert!(para.contains("fully mobilized"));
        assert!(para.contains("alliance with Country B"));
        assert!(para.contains("at war with Country C"));
    }

    #[test]
    fn translate_handles_no_relations() {
        let b = Board::new(AgentId(0));
        let stick = Stick::default();
        let para = translate(&b, &stick, &names());
        assert!(para.contains("no active treaties"));
    }
}
