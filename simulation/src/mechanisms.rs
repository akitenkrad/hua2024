//! socsim フレームワーク上の WarAgent 外交メカニズム (5 Mechanism × 6 phase)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! 行動の伝播・解決・エスカレーション・Board 更新を `ctx`/Board ベースで決定論的に
//! 行い，上層 (非決定的 LLM レイヤ) は [`WarClient`] (キャッシュ付き Ollama→OpenAI
//! フォールバック) 越しの «4 ステップ誘導推論 + 秘書検証» を行う．
//!
//! 論文のラウンド処理を 6-phase へ割り当てる:
//!
//! | Mechanism | Phase | 役割 |
//! |-----------|-------|------|
//! | [`SituationMechanism`]       | Environment | Board/Stick を Translate して文脈提示・トリガー注入 (round 0) |
//! | [`CountryDecisionMechanism`] | Decision    | **各国 LLM が 4 ステップ推論 + 秘書検証で行動決定** (LLM 所在) |
//! | [`DiplomacyMechanism`]       | Interaction | publicity 伝播・同盟↔受諾・宣戦の解決 (Board 更新) |
//! | [`EscalationMechanism`]      | Interaction | エスカレーション (同盟国の参戦) |
//! | [`BoardUpdateMechanism`]     | PostStep    | Stick(MO)/Board 確定・イベントログ・収束判定 (request_stop) |
//!
//! LLM 呼び出しは Decision の 1 mechanism に閉じ込める (= 1 ラウンドあたり
//! 国数 × (1 + secretary_passes) 回)．他はすべて LLM 非依存・決定論的である．
//! LLM クライアントとメタデータ・指標バッファは `Rc<RefCell<…>>` で共有する．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::board::{translate, Relation};
use crate::config::LlmSettings;
use crate::llm::{llm_config, WarClient};
use crate::metrics::{compute_round_metric, RoundMetric};
use crate::prompts::{
    decision_prompt, parse_action, secretary_prompt, DecisionContext, ParsedAction,
};
use crate::world::{Action, ActionKind, Event, Publicity, WarWorld};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<WarClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;
/// 共有 ラウンド指標バッファ (ドライバが run 後に CSV へ書き出す)．
pub type SharedMetrics = Rc<RefCell<Vec<RoundMetric>>>;

// scratch キー．同一ステップ内でフェーズ間に値を受け渡す．
const SCRATCH_CONTEXT: &str = "board_context"; // BTreeMap<u64, String>
const SCRATCH_WAR_JOINS: &str = "war_joins"; // Vec<(u64 actor, u64 target)>: escalation の参戦

// =========================================================================== //
// 1. SituationMechanism (Environment)
// =========================================================================== //

/// 各国の Board/Stick を Translate して文脈パラグラフを用意し，round 0 では
/// トリガーイベントを注入する (`Environment` フェーズ; LLM 非依存)．
///
/// 文脈パラグラフは scratch に «国 raw id → パラグラフ» で置き，Decision が
/// プロンプトに使う．`pending_actions` をこのラウンドのために空にする．
pub struct SituationMechanism;

impl Mechanism<WarWorld> for SituationMechanism {
    fn name(&self) -> &str {
        "situation"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, WarWorld>) -> Result<()> {
        let round = ctx.world.current_round();
        ctx.world.round = round;
        ctx.world.pending_actions.clear();

        let names = ctx.world.name_map();
        let mut contexts: BTreeMap<u64, String> = BTreeMap::new();
        for (id, country) in &ctx.world.countries {
            let board = ctx
                .world
                .boards
                .get(id)
                .cloned()
                .unwrap_or_else(|| crate::board::Board::new(*id));
            let para = translate(&board, &country.stick, &names);
            contexts.insert(id.0, para);
        }
        ctx.scratch.insert(SCRATCH_CONTEXT, contexts);

        // round 0: トリガー注入 (まだ注入していなければ)．
        if round == 0 && !ctx.world.trigger_injected {
            ctx.world.trigger_injected = true;
        }
        Ok(())
    }
}

// =========================================================================== //
// 2. CountryDecisionMechanism (Decision, LLM)
// =========================================================================== //

/// 各国 LLM が 4 ステップ誘導推論 + 秘書検証で行動を決定する (`Decision` フェーズ;
/// **LLM 所在は唯一ここ**)．
///
/// 各国について «decision_prompt → parse → (secretary_prompt → parse) × passes» を
/// 実行し，最終 [`ParsedAction`] を world の `pending_actions` に蓄積する (この時点で
/// Board は不変; 同期更新)．LLM 呼び出し回数は 国数 × (1 + secretary_passes)．
pub struct CountryDecisionMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    secretary_passes: usize,
}

impl CountryDecisionMechanism {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        secretary_passes: usize,
    ) -> Self {
        CountryDecisionMechanism {
            client,
            metadata,
            settings,
            secretary_passes,
        }
    }

    /// 1 国の行動を LLM で決定する (4 ステップ + 秘書検証)．
    fn decide_one(
        &self,
        ctx: &mut StepContext<'_, WarWorld>,
        actor: AgentId,
        board_context: String,
    ) -> Result<ParsedAction> {
        let round = ctx.world.round;
        let names = ctx.world.name_map();
        let country = ctx
            .world
            .countries
            .get(&actor)
            .expect("actor exists")
            .clone();

        // 対象候補 = 自国以外の全国．
        let targets: Vec<(AgentId, String)> = ctx
            .world
            .countries
            .keys()
            .filter(|id| **id != actor)
            .map(|id| (*id, names.get(id).cloned().unwrap_or_default()))
            .collect();
        let valid_ids: Vec<AgentId> = targets.iter().map(|(id, _)| *id).collect();

        // inbox 要約 (直近のみ)．
        let inbox_summary: Vec<String> = ctx
            .world
            .inbox
            .get(&actor)
            .map(|acts| {
                acts.iter()
                    .rev()
                    .take(5)
                    .map(|a| describe_action(a, &names))
                    .collect()
            })
            .unwrap_or_default();

        // round 0 のみトリガーを与える．
        let trigger = if round == 0 {
            ctx.world.trigger.as_deref()
        } else {
            None
        };

        let cx = DecisionContext {
            actor,
            country: &country,
            board_context,
            trigger,
            inbox_summary,
            targets,
            round,
        };

        // --- 4 ステップ誘導推論 ---
        let prompt = decision_prompt(&cx);
        let text = self.call(&prompt)?;
        let mut action = parse_action(&text, &valid_ids);

        // --- 秘書検証 (configurable passes; 既定 1) ---
        for _ in 0..self.secretary_passes {
            let proposed = serde_json::json!({
                "action": action.kind.label(),
                "target": action.target,
                "publicity": action.publicity.label(),
            })
            .to_string();
            let sprompt = secretary_prompt(&cx, &proposed);
            let stext = self.call(&sprompt)?;
            action = parse_action(&stext, &valid_ids);
        }

        Ok(action)
    }

    /// LLM を 1 回呼び，メタデータを記録して本文を返す．
    fn call(&self, prompt: &str) -> Result<String> {
        let mut client = self.client.borrow_mut();
        let resp = client
            .complete(prompt, &llm_config(&self.settings))
            .map_err(|e| {
                SocsimError::Mechanism(format!("country decision LLM call failed: {e}"))
            })?;
        self.metadata.borrow_mut().record(resp.metadata.clone());
        Ok(resp.text)
    }
}

impl Mechanism<WarWorld> for CountryDecisionMechanism {
    fn name(&self) -> &str {
        "country_decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, WarWorld>) -> Result<()> {
        let round = ctx.world.round;
        let contexts: BTreeMap<u64, String> = ctx
            .scratch
            .get::<BTreeMap<u64, String>>(SCRATCH_CONTEXT)
            .cloned()
            .unwrap_or_default();

        // 国 id をソート順に決定する (決定論; LLM はキャッシュで擬似決定論)．
        let actor_ids: Vec<AgentId> = ctx.world.countries.keys().copied().collect();
        let mut actions: Vec<Action> = Vec::with_capacity(actor_ids.len());
        for actor in actor_ids {
            let board_ctx = contexts.get(&actor.0).cloned().unwrap_or_default();
            let parsed = self.decide_one(ctx, actor, board_ctx)?;
            actions.push(Action {
                actor,
                kind: parsed.kind,
                target: parsed.target.map(AgentId),
                publicity: parsed.publicity,
                round,
            });
        }

        ctx.world.pending_actions = actions;
        Ok(())
    }
}

// =========================================================================== //
// 3. DiplomacyMechanism (Interaction)
// =========================================================================== //

/// publicity に基づく伝播と，同盟要請↔受諾・宣戦布告・条約・和平の解決
/// (`Interaction` フェーズ; LLM 非依存)．
///
/// pending_actions を一括解決する (同期更新):
/// - 配送: 公開行動は全国の inbox へ，秘密行動は対象国の inbox のみへ．
/// - 宣戦布告 (W): 双方の Board を War に (片務宣戦も交戦扱い)．
/// - 同盟要請 (Alliance): 同ラウンドに対象国も自国へ Alliance/Message を返していれば
///   «相互合意» として双方 Board を Alliance に (簡略受諾モデル)．それ以外は申し出を
///   inbox に残すのみ (翌ラウンドに反応しうる)．
/// - 不干渉 (NonAggression): 同様に相互提案で双方 T．片務でも対象視点 T を緩く張る
///   (緊張緩和の意図を尊重)．
/// - 和平 (Peace): 双方 Board を Peace に戻す (交戦解消)．
/// - 総動員 (Mobilize): EscalationMechanism と BoardUpdate で Stick に反映する
///   (ここでは伝播のみ)．
pub struct DiplomacyMechanism;

impl Mechanism<WarWorld> for DiplomacyMechanism {
    fn name(&self) -> &str {
        "diplomacy"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, WarWorld>) -> Result<()> {
        let actions = ctx.world.pending_actions.clone();

        // --- 配送 (publicity 伝播) ---
        let all_ids: Vec<AgentId> = ctx.world.countries.keys().copied().collect();
        for act in &actions {
            match act.publicity {
                Publicity::Public => {
                    for &id in &all_ids {
                        if id != act.actor {
                            ctx.world.inbox.entry(id).or_default().push(act.clone());
                        }
                    }
                }
                Publicity::Secret => {
                    if let Some(t) = act.target {
                        ctx.world.inbox.entry(t).or_default().push(act.clone());
                    }
                }
            }
        }

        // 相互提案の検出: 「a→b の Alliance/NonAggression」に対し「b→a の
        // 同種 or Message」が同ラウンドにあれば合意とみなす．
        let proposes = |kind: ActionKind, from: AgentId, to: AgentId| -> bool {
            actions.iter().any(|x| {
                x.actor == from
                    && x.target == Some(to)
                    && (x.kind == kind || x.kind == ActionKind::Message)
            })
        };

        // --- 解決 ---
        for act in &actions {
            let actor = act.actor;
            let target = act.target;
            match act.kind {
                ActionKind::DeclareWar => {
                    if let Some(t) = target {
                        set_relation(ctx, actor, t, Relation::War);
                        set_relation(ctx, t, actor, Relation::War);
                    }
                }
                ActionKind::Alliance => {
                    if let Some(t) = target {
                        if proposes(ActionKind::Alliance, t, actor) {
                            set_relation(ctx, actor, t, Relation::Alliance);
                            set_relation(ctx, t, actor, Relation::Alliance);
                        }
                    }
                }
                ActionKind::NonAggression => {
                    if let Some(t) = target {
                        // 片務でも対象視点に T を張る (緊張緩和)．相互なら双方 T．
                        set_relation(ctx, actor, t, Relation::NonAggression);
                        if proposes(ActionKind::NonAggression, t, actor) {
                            set_relation(ctx, t, actor, Relation::NonAggression);
                        }
                    }
                }
                ActionKind::Peace => {
                    if let Some(t) = target {
                        set_relation(ctx, actor, t, Relation::Peace);
                        set_relation(ctx, t, actor, Relation::Peace);
                    }
                }
                // Mobilize は BoardUpdate で Stick に反映．Wait/Message は関係不変．
                _ => {}
            }
        }
        Ok(())
    }
}

/// 国 `owner` の Board に «owner→other = rel» を書く (Board が無ければ作る)．
fn set_relation(
    ctx: &mut StepContext<'_, WarWorld>,
    owner: AgentId,
    other: AgentId,
    rel: Relation,
) {
    ctx.world
        .boards
        .entry(owner)
        .or_insert_with(|| crate::board::Board::new(owner))
        .set_relation(other, rel);
}

// =========================================================================== //
// 4. EscalationMechanism (Interaction)
// =========================================================================== //

/// エスカレーション動学: 同盟国が交戦に巻き込まれる (`Interaction` フェーズ;
/// LLM 非依存; Diplomacy の後に走る)．
///
/// «A が B に宣戦» したとき，B の同盟国 (B の Board で Alliance) は A に対して
/// 自動参戦 (War) する (1 ホップ伝播)．これを scratch に記録し，BoardUpdate が
/// イベントログに残す．保守的国の参戦も今回は «同盟義務» として一律に発火する
/// (簡略モデル; スタンス依存は拡張点)．
pub struct EscalationMechanism;

impl Mechanism<WarWorld> for EscalationMechanism {
    fn name(&self) -> &str {
        "escalation"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, WarWorld>) -> Result<()> {
        // このラウンドの新規宣戦布告 (actor→target = War) を集める．
        let new_wars: Vec<(AgentId, AgentId)> = ctx
            .world
            .pending_actions
            .iter()
            .filter(|a| a.kind == ActionKind::DeclareWar)
            .filter_map(|a| a.target.map(|t| (a.actor, t)))
            .collect();

        let mut joins: Vec<(u64, u64)> = Vec::new();
        for (aggressor, victim) in new_wars {
            // victim の同盟国を取得 (victim の Board の M)．
            let allies = ctx.world.counterparts(victim, Relation::Alliance);
            for ally in allies {
                if ally == aggressor {
                    continue;
                }
                // ally と aggressor が既に War でなければ参戦．
                let already = ctx
                    .world
                    .boards
                    .get(&ally)
                    .map(|b| b.relation_to(aggressor) == Relation::War)
                    .unwrap_or(false);
                if !already {
                    set_relation(ctx, ally, aggressor, Relation::War);
                    set_relation(ctx, aggressor, ally, Relation::War);
                    joins.push((ally.0, aggressor.0));
                }
            }
        }

        ctx.scratch.insert(SCRATCH_WAR_JOINS, joins);
        Ok(())
    }
}

// =========================================================================== //
// 5. BoardUpdateMechanism (PostStep)
// =========================================================================== //

/// Stick (MO) 確定・イベントログ追記・指標計算・収束判定 (`PostStep` フェーズ;
/// LLM 非依存)．
///
/// - 総動員 (Mobilize) 行動を出した国の Stick.mobilized を立てる．交戦に入った国も
///   総動員する (戦時動員)．
/// - pending_actions と escalation の参戦をイベントログへ追記する．
/// - ラウンド指標を計算して共有バッファへ push する．
/// - 収束判定: 世界大戦勃発 (W 対 >= war_threshold) または最終ラウンドで
///   `ctx.request_stop()`．
pub struct BoardUpdateMechanism {
    rounds: u64,
    war_threshold: usize,
    metrics: SharedMetrics,
}

impl BoardUpdateMechanism {
    pub fn new(rounds: u64, war_threshold: usize, metrics: SharedMetrics) -> Self {
        BoardUpdateMechanism {
            rounds,
            war_threshold,
            metrics,
        }
    }
}

impl Mechanism<WarWorld> for BoardUpdateMechanism {
    fn name(&self) -> &str {
        "board_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, WarWorld>) -> Result<()> {
        let round = ctx.world.round;
        let names = ctx.world.name_map();
        let actions = ctx.world.pending_actions.clone();

        // --- Stick (MO) 更新 ---
        for act in &actions {
            if act.kind == ActionKind::Mobilize {
                if let Some(c) = ctx.world.countries.get_mut(&act.actor) {
                    c.stick.mobilized = true;
                }
            }
        }
        // 交戦に入った国は総動員する (戦時動員)．
        let at_war: Vec<AgentId> = ctx
            .world
            .boards
            .iter()
            .filter(|(_, b)| !b.counterparts(Relation::War).is_empty())
            .map(|(id, _)| *id)
            .collect();
        for id in at_war {
            if let Some(c) = ctx.world.countries.get_mut(&id) {
                c.stick.mobilized = true;
            }
        }

        // --- イベントログ (行動) ---
        for act in &actions {
            ctx.world.event_log.push(Event {
                round,
                actor: act.actor.0,
                action: act.kind.label().to_string(),
                target: act.target.map(|t| t.0),
                publicity: act.publicity.label().to_string(),
            });
        }
        // --- イベントログ (escalation 参戦) ---
        let joins: Vec<(u64, u64)> = ctx
            .scratch
            .get::<Vec<(u64, u64)>>(SCRATCH_WAR_JOINS)
            .cloned()
            .unwrap_or_default();
        for (ally, aggressor) in joins {
            ctx.world.event_log.push(Event {
                round,
                actor: ally,
                action: "escalate_join_war".to_string(),
                target: Some(aggressor),
                publicity: Publicity::Public.label().to_string(),
            });
        }

        // --- 各国 memory 更新 (inbox の直近を要約として残す) ---
        let ids: Vec<AgentId> = ctx.world.countries.keys().copied().collect();
        for id in ids {
            let summary: Vec<String> = ctx
                .world
                .inbox
                .get(&id)
                .map(|acts| {
                    acts.iter()
                        .rev()
                        .take(3)
                        .map(|a| describe_action(a, &names))
                        .collect()
                })
                .unwrap_or_default();
            if let Some(c) = ctx.world.countries.get_mut(&id) {
                for line in summary {
                    c.memory.push(line);
                }
                if c.memory.len() > 10 {
                    let excess = c.memory.len() - 10;
                    c.memory.drain(0..excess);
                }
            }
        }

        // --- ラウンド指標 ---
        let metric = compute_round_metric(ctx.world, round, self.war_threshold);
        let outbreak = metric.war_outbreak == 1;
        ctx.scratch.insert("war_outbreak", outbreak);
        ctx.scratch.insert("n_conflicts", metric.n_conflicts);
        self.metrics.borrow_mut().push(metric);

        // --- 収束判定 ---
        let is_last_round = round + 1 >= self.rounds;
        if outbreak || is_last_round {
            ctx.request_stop();
        }
        Ok(())
    }
}

/// 行動を人間可読な 1 行に要約する (inbox / memory / ログ用)．
fn describe_action(act: &Action, names: &BTreeMap<AgentId, String>) -> String {
    let actor = names
        .get(&act.actor)
        .cloned()
        .unwrap_or_else(|| format!("Country#{}", act.actor.0));
    match act.target {
        Some(t) => {
            let target = names
                .get(&t)
                .cloned()
                .unwrap_or_else(|| format!("Country#{}", t.0));
            format!(
                "{actor} -> {target}: {} ({})",
                act.kind.label(),
                act.publicity.label()
            )
        }
        None => format!("{actor}: {} ({})", act.kind.label(), act.publicity.label()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Country, Profile, Stance};

    #[test]
    fn describe_action_formats() {
        let mut names = BTreeMap::new();
        names.insert(AgentId(0), "Country A".to_string());
        names.insert(AgentId(1), "Country B".to_string());
        let act = Action {
            actor: AgentId(0),
            kind: ActionKind::Alliance,
            target: Some(AgentId(1)),
            publicity: Publicity::Public,
            round: 0,
        };
        let s = describe_action(&act, &names);
        assert!(s.contains("Country A -> Country B: alliance (public)"));
        let _ = Country::new("x", Profile::default(), Stance::Neutral);
    }
}
