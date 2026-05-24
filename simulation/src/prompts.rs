//! LLM プロンプト生成と応答パース．
//!
//! WarAgent の意思決定プロンプトを構築する:
//! 1. **国の 4 ステップ誘導推論** (`CountryDecisionMechanism`): プロフィール +
//!    Translate された Board/Stick 文脈 + トリガー + inbox を提示し，«同盟候補特定
//!    → 敵対候補認識 → 推奨行動 → 最終行動 (7 行動空間)» を JSON で求める．
//! 2. **秘書検証** (同 Mechanism): 国の行動案を «書式・対象妥当性» の観点で検査・
//!    修正し，妥当な行動 JSON を返す (configurable secretary_passes 回)．
//!
//! 応答パースは「まず JSON として読む → 失敗時はフォールバック (Wait)」の二段で
//! 頑健化する (ローカルモデルは厳密 JSON を返さないことがある)．

use std::collections::BTreeMap;

use socsim_core::AgentId;

use crate::world::{ActionKind, Country, Publicity};

/// 1 国の意思決定プロンプトに渡す «周辺文脈»．
pub struct DecisionContext<'a> {
    /// この国の `AgentId`．
    pub actor: AgentId,
    /// この国の状態．
    pub country: &'a Country,
    /// Translate 済みの Board/Stick 文脈パラグラフ．
    pub board_context: String,
    /// 第 1 ラウンドのトリガー説明 (なければ None)．
    pub trigger: Option<&'a str>,
    /// 直近に受信したメッセージ/行動の要約 (inbox; 公開 or 自分宛)．
    pub inbox_summary: Vec<String>,
    /// 選択可能な対象国 (id, 匿名名) のリスト (自国を除く)．
    pub targets: Vec<(AgentId, String)>,
    /// 現在のラウンド (0 始まり)．
    pub round: u64,
}

/// 国の 4 ステップ誘導推論プロンプトを構築する．
///
/// プロフィール 6 次元・スタンス・Translate 文脈・トリガー・inbox・選択可能対象を
/// 提示し，4 ステップ (allies → adversaries → recommended → final) を経て最終行動を
/// JSON で答えさせる．対象は «id» で指定させる (匿名名と併記)．
pub fn decision_prompt(cx: &DecisionContext<'_>) -> String {
    let c = cx.country;
    let mut s = String::new();
    s.push_str(
        "You are the leadership of a country in a multi-country international crisis. You must \
         decide this round's single action that best serves your national interest.\n\n",
    );

    // --- 6 次元プロフィール + スタンス ---
    s.push_str(&format!("## Your country: {}\n", c.name));
    s.push_str(&format!("- Leadership: {}\n", c.profile.leadership));
    s.push_str(&format!("- Military capability: {}\n", c.profile.military));
    s.push_str(&format!("- Resources: {}\n", c.profile.resources));
    s.push_str(&format!("- Historical background: {}\n", c.profile.history));
    s.push_str(&format!("- Key policy: {}\n", c.profile.policy));
    s.push_str(&format!("- Public morale: {}\n", c.profile.morale));
    s.push_str(&format!("- Stance: {}\n", c.stance.describe()));

    // --- Board/Stick 文脈 (Translate) ---
    s.push_str("\n## Current situation\n");
    s.push_str(&cx.board_context);

    // --- トリガー (round 0 のみ与えられる) ---
    if let Some(t) = cx.trigger {
        s.push_str(&format!("\n## Breaking event this round\n{t}\n"));
    }

    // --- inbox ---
    if !cx.inbox_summary.is_empty() {
        s.push_str("\n## Recent diplomatic signals you have received\n");
        for line in &cx.inbox_summary {
            s.push_str("- ");
            s.push_str(line);
            s.push('\n');
        }
    }

    // --- 選択可能対象 ---
    s.push_str("\n## Other countries (target ids)\n");
    for (id, name) in &cx.targets {
        s.push_str(&format!("- id {}: {}\n", id.0, name));
    }

    // --- 4 ステップ誘導 + 行動空間 + 出力形式 ---
    s.push_str(&format!(
        "\n## Reasoning (round {})\n\
         Think step by step, then act:\n\
         1. Identify which countries are your natural allies.\n\
         2. Recognize which countries are your adversaries or threats.\n\
         3. Recommend the action that best advances your interest.\n\
         4. Commit to ONE final action from the action space below.\n\n\
         ## Action space\n\
         - \"wait\": do nothing this round.\n\
         - \"mobilize\": order full mobilization (no target).\n\
         - \"declare_war\": declare war on a target.\n\
         - \"alliance\": propose a military alliance to a target.\n\
         - \"non_aggression\": propose a non-aggression treaty to a target.\n\
         - \"peace\": offer peace to a target you are at war with.\n\
         - \"message\": send a diplomatic message to a target (no relation change).\n\n\
         ## Output\n\
         Answer with JSON only, e.g. {{\"action\": \"alliance\", \"target\": 3, \"publicity\": \"public\"}}.\n\
         For \"wait\"/\"mobilize\" omit target. \"publicity\" is \"public\" or \"secret\" (default public).\n",
        cx.round
    ));
    s
}

/// 秘書検証プロンプトを構築する．
///
/// 国の行動案 (JSON 文字列) を提示し，«行動が 7 行動空間にあり，対象が有効な
/// 他国 id か» を検査・修正させ，妥当な行動 JSON «のみ» を返させる．
pub fn secretary_prompt(cx: &DecisionContext<'_>, proposed: &str) -> String {
    let mut s = String::new();
    s.push_str(
        "You are the secretary verifying your country's proposed diplomatic action for format and \
         logical validity before it is dispatched.\n\n",
    );
    s.push_str(&format!("## Country: {}\n", cx.country.name));
    s.push_str("## Valid target ids\n");
    for (id, name) in &cx.targets {
        s.push_str(&format!("- id {}: {}\n", id.0, name));
    }
    s.push_str(&format!("\n## Proposed action\n{proposed}\n"));
    s.push_str(
        "\n## Task\n\
         Verify the action is one of wait/mobilize/declare_war/alliance/non_aggression/peace/message, \
         that target-bearing actions name a valid other-country id, and that wait/mobilize have no \
         target. If valid, return it unchanged. If invalid, correct it (e.g. fall back to \"wait\").\n\
         Answer with JSON only, same schema: {\"action\": ..., \"target\": ..., \"publicity\": ...}.\n",
    );
    s
}

/// パース済みの最終行動 (対象は raw id)．
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedAction {
    pub kind: ActionKind,
    /// 対象国の raw u64 (対象不要行動では None)．
    pub target: Option<u64>,
    pub publicity: Publicity,
}

impl ParsedAction {
    /// 既定のフォールバック (Wait/public)．
    pub fn wait() -> Self {
        ParsedAction {
            kind: ActionKind::Wait,
            target: None,
            publicity: Publicity::Public,
        }
    }
}

/// 行動応答をパースする．有効な対象 id 集合 `valid_targets` で対象を検証する．
///
/// 1. JSON `{"action","target","publicity"}` を読む．
/// 2. action が対象を要するのに target が無効 → Wait にフォールバック．
/// 3. JSON 不能 → 本文のキーワードから推測，それも無理なら Wait．
pub fn parse_action(text: &str, valid_targets: &[AgentId]) -> ParsedAction {
    let valid: BTreeMap<u64, ()> = valid_targets.iter().map(|a| (a.0, ())).collect();

    if let Some(json) = extract_json_object(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            if let Some(obj) = v.as_object() {
                let action_str = obj
                    .get("action")
                    .or_else(|| obj.get("final_action"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase();
                let kind = action_kind_from_str(&action_str);
                let publicity = obj
                    .get("publicity")
                    .and_then(|x| x.as_str())
                    .map(|p| {
                        if p.trim().eq_ignore_ascii_case("secret") {
                            Publicity::Secret
                        } else {
                            Publicity::Public
                        }
                    })
                    .unwrap_or(Publicity::Public);
                let target = obj.get("target").and_then(|x| {
                    x.as_u64()
                        .or_else(|| x.as_str().and_then(|s| s.trim().parse().ok()))
                });

                if let Some(kind) = kind {
                    return finalize(kind, target, publicity, &valid);
                }
            }
        }
    }

    // 本文からキーワード推測 (JSON 失敗時)．
    let lower = text.to_ascii_lowercase();
    let kind = if lower.contains("declare_war") || lower.contains("declare war") {
        Some(ActionKind::DeclareWar)
    } else if lower.contains("alliance") {
        Some(ActionKind::Alliance)
    } else if lower.contains("non_aggression") || lower.contains("non-aggression") {
        Some(ActionKind::NonAggression)
    } else if lower.contains("mobilize") || lower.contains("mobilise") {
        Some(ActionKind::Mobilize)
    } else if lower.contains("peace") {
        Some(ActionKind::Peace)
    } else if lower.contains("message") {
        Some(ActionKind::Message)
    } else if lower.contains("wait") {
        Some(ActionKind::Wait)
    } else {
        None
    };
    match kind {
        Some(k) => {
            // 本文ルートでは対象を取り出せないことが多いので，最初の数字を拾う．
            let target = first_number(text);
            finalize(k, target, Publicity::Public, &valid)
        }
        None => ParsedAction::wait(),
    }
}

/// 対象妥当性を確定する (対象必須なのに無効 → Wait)．
fn finalize(
    kind: ActionKind,
    target: Option<u64>,
    publicity: Publicity,
    valid: &BTreeMap<u64, ()>,
) -> ParsedAction {
    if kind.needs_target() {
        match target {
            Some(t) if valid.contains_key(&t) => ParsedAction {
                kind,
                target: Some(t),
                publicity,
            },
            // 対象が無効 → 行動を取り消して Wait．
            _ => ParsedAction::wait(),
        }
    } else {
        ParsedAction {
            kind,
            target: None,
            publicity,
        }
    }
}

/// 行動文字列を [`ActionKind`] へ．
fn action_kind_from_str(s: &str) -> Option<ActionKind> {
    match s {
        "wait" => Some(ActionKind::Wait),
        "mobilize" | "mobilise" => Some(ActionKind::Mobilize),
        "declare_war" | "declarewar" | "war" => Some(ActionKind::DeclareWar),
        "alliance" | "ally" => Some(ActionKind::Alliance),
        "non_aggression" | "nonaggression" | "non-aggression" | "treaty" => {
            Some(ActionKind::NonAggression)
        }
        "peace" => Some(ActionKind::Peace),
        "message" | "msg" => Some(ActionKind::Message),
        _ => None,
    }
}

/// 文字列から最初の `{ … }` ブロックを切り出す．
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    Some(text[start..=end].to_string())
}

/// 本文中の最初の非負整数を拾う．
fn first_number(text: &str) -> Option<u64> {
    for tok in text.split(|c: char| !c.is_ascii_digit()) {
        if let Ok(k) = tok.parse::<u64>() {
            return Some(k);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> Vec<AgentId> {
        vec![AgentId(1), AgentId(2), AgentId(3)]
    }

    #[test]
    fn parses_alliance_json() {
        let a = parse_action(
            "{\"action\": \"alliance\", \"target\": 3, \"publicity\": \"public\"}",
            &valid(),
        );
        assert_eq!(a.kind, ActionKind::Alliance);
        assert_eq!(a.target, Some(3));
        assert_eq!(a.publicity, Publicity::Public);
    }

    #[test]
    fn secret_declare_war_parsed() {
        let a = parse_action(
            "{\"action\": \"declare_war\", \"target\": 1, \"publicity\": \"secret\"}",
            &valid(),
        );
        assert_eq!(a.kind, ActionKind::DeclareWar);
        assert_eq!(a.publicity, Publicity::Secret);
    }

    #[test]
    fn mobilize_needs_no_target() {
        let a = parse_action("{\"action\": \"mobilize\"}", &valid());
        assert_eq!(a.kind, ActionKind::Mobilize);
        assert_eq!(a.target, None);
    }

    #[test]
    fn invalid_target_falls_back_to_wait() {
        // target 9 is not a valid country id.
        let a = parse_action("{\"action\": \"alliance\", \"target\": 9}", &valid());
        assert_eq!(a.kind, ActionKind::Wait);
    }

    #[test]
    fn garbage_falls_back_to_wait() {
        let a = parse_action("I am unsure what to do.", &valid());
        assert_eq!(a.kind, ActionKind::Wait);
    }

    #[test]
    fn prose_keyword_fallback() {
        let a = parse_action("We should declare war on country 2.", &valid());
        assert_eq!(a.kind, ActionKind::DeclareWar);
        assert_eq!(a.target, Some(2));
    }
}
