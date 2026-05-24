//! 評価指標 (論文 §6 数式 / Table 2)．
//!
//! - **alliance_mi**: 同盟分割の相互情報量 (vs 史実分割)．同盟は推移的なので国家
//!   集合の «分割» とみなし，シミュレート分割 U と史実分割 V の MI を計算する．
//! - **declaration_jaccard**: 宣戦布告 (国家対集合) の Jaccard (vs 史実 W 対集合)．
//! - **mobilization_jaccard**: 総動員 (単集合) の Jaccard (vs 史実総動員国集合)．
//! - **war_outbreak / escalation_round / n_conflicts / cold_war_flag**: マクロ判定．
//!
//! 史実参照集合 (WWI) は匿名 id (A=0..H=7) で定数として保持する．MI/Jaccard は
//! Rust 側で計算し ([0,1] に正規化)，Python (scipy) への依存は不要にする．

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use socsim_net::SocialNetwork;

use crate::board::Relation;
use crate::world::WarWorld;

// --------------------------------------------------------------------------- //
// 史実参照集合 (WWI; 匿名 id A=0..H=7)
// --------------------------------------------------------------------------- //

/// 史実 WWI 同盟分割 (id のグループ; 推移的同盟クラスタ)．
///
/// 中央同盟系 {A,B,G} = {0,1,6}，協商系 {C,D,E,H} = {2,3,4,7}，中立 {F} = {5}
/// (F=Italy は当初中立)．
pub fn historical_alliance_partition() -> Vec<Vec<u64>> {
    vec![vec![0, 1, 6], vec![2, 3, 4, 7], vec![5]]
}

/// 史実 WWI 宣戦布告対集合 (無向対; おおまかな主要交戦)．
///
/// A-C, A-D, A-E, B-E, B-H, G-E など中央 vs 協商の主要対を取る (例示的近似)．
pub fn historical_war_pairs() -> BTreeSet<(u64, u64)> {
    [
        (0, 2), // A-C
        (0, 3), // A-D
        (0, 4), // A-E
        (1, 4), // B-E
        (1, 7), // B-H
        (4, 6), // E-G
    ]
    .into_iter()
    .map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
    .collect()
}

/// 史実 WWI 総動員国集合 (主要参戦国はおおむね総動員したとみなす)．
pub fn historical_mobilized() -> BTreeSet<u64> {
    [0u64, 1, 2, 3, 4, 6, 7].into_iter().collect()
}

// --------------------------------------------------------------------------- //
// 数式 (MI / Jaccard)
// --------------------------------------------------------------------------- //

/// 2 つの分割 (クラスタリング) の相互情報量スコア．
///
/// $\mathrm{MI}(U,V)=\sum_i\sum_j \frac{|U_i\cap V_j|}{N}\log\frac{N|U_i\cap V_j|}{|U_i||V_j|}$．
/// 全要素の和集合に現れる N 個の要素で正規化する．本実装では «どちらの分割にも
/// 現れる要素» の合併を母集合 N とし，対数は自然対数．戻り値は «正規化相互情報量
/// NMI = MI / sqrt(H(U) H(V))» で [0,1] に収める (両エントロピー 0 のときは 1.0)．
pub fn alliance_mi(sim: &[Vec<u64>], hist: &[Vec<u64>]) -> f64 {
    // 母集合 = 両分割に現れる要素の合併．
    let mut universe: BTreeSet<u64> = BTreeSet::new();
    for g in sim.iter().chain(hist.iter()) {
        for &x in g {
            universe.insert(x);
        }
    }
    let n = universe.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;

    // 各クラスタを母集合に制限した集合へ．
    let restrict = |groups: &[Vec<u64>]| -> Vec<BTreeSet<u64>> {
        groups
            .iter()
            .map(|g| g.iter().copied().filter(|x| universe.contains(x)).collect())
            .filter(|s: &BTreeSet<u64>| !s.is_empty())
            .collect()
    };
    let u = restrict(sim);
    let v = restrict(hist);

    // MI．
    let mut mi = 0.0;
    for ui in &u {
        for vj in &v {
            let inter = ui.intersection(vj).count();
            if inter == 0 {
                continue;
            }
            let p_ij = inter as f64 / nf;
            let p_i = ui.len() as f64 / nf;
            let p_j = vj.len() as f64 / nf;
            mi += p_ij * (p_ij / (p_i * p_j)).ln();
        }
    }

    // エントロピー (正規化用)．
    let entropy = |groups: &[BTreeSet<u64>]| -> f64 {
        let mut h = 0.0;
        for g in groups {
            let p = g.len() as f64 / nf;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    };
    let hu = entropy(&u);
    let hv = entropy(&v);
    let denom = (hu * hv).sqrt();
    if denom < 1e-12 {
        // 両者とも単一クラスタ (完全合意) → NMI = 1．
        1.0
    } else {
        (mi / denom).clamp(0.0, 1.0)
    }
}

/// 2 集合の Jaccard 指標 |A∩B| / |A∪B| ∈ [0,1]．両空なら 1 (完全一致扱い)．
pub fn jaccard<T: Ord>(a: &BTreeSet<T>, b: &BTreeSet<T>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

// --------------------------------------------------------------------------- //
// World からの抽出
// --------------------------------------------------------------------------- //

/// Board の同盟関係 (M) から «同盟クラスタ分割» を導出する (推移閉包; union-find)．
///
/// 同盟 (M) を無向辺とみなし連結成分を取る．同盟に属さない国も自身単独の
/// クラスタとして含める (母集合を全国にする)．
pub fn alliance_partition(world: &WarWorld) -> Vec<Vec<u64>> {
    let ids: Vec<u64> = world.countries.keys().map(|a| a.0).collect();
    let idx: BTreeMap<u64, usize> = ids.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let mut parent: Vec<usize> = (0..ids.len()).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        // 経路圧縮．
        let mut c = x;
        while parent[c] != r {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }

    for (owner, board) in &world.boards {
        for other in board.counterparts(Relation::Alliance) {
            if let (Some(&i), Some(&j)) = (idx.get(&owner.0), idx.get(&other.0)) {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut groups: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    for (i, &id) in ids.iter().enumerate() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(id);
    }
    groups.into_values().collect()
}

/// Board の同盟関係 (M) から `socsim-net` のグローバル同盟グラフ (無向) を構築する．
///
/// Board が **source of truth** であり，この `SocialNetwork` はそこから導出される
/// «ビュー» である (可視化・連結成分の確認用)．[`alliance_partition`] の union-find
/// と [`SocialNetwork::connected_components`] は同じクラスタ数を返す (相互検算)．
pub fn alliance_network(world: &WarWorld) -> SocialNetwork {
    let mut net = SocialNetwork::empty();
    for id in world.countries.keys() {
        net.add_node(*id);
    }
    for (owner, board) in &world.boards {
        for other in board.counterparts(Relation::Alliance) {
            // 無向グラフなので owner < other のときだけ張る (重複辺を避ける)．
            if owner.0 < other.0 && world.countries.contains_key(&other) {
                net.add_edge(*owner, other);
            }
        }
    }
    net
}

/// Board の宣戦布告 (W) から無向対集合を導出する．
pub fn war_pair_set(world: &WarWorld) -> BTreeSet<(u64, u64)> {
    let mut set = BTreeSet::new();
    for (owner, board) in &world.boards {
        for other in board.counterparts(Relation::War) {
            let (a, b) = if owner.0 <= other.0 {
                (owner.0, other.0)
            } else {
                (other.0, owner.0)
            };
            set.insert((a, b));
        }
    }
    set
}

/// 総動員した国の集合 (Stick: MO)．
pub fn mobilized_set(world: &WarWorld) -> BTreeSet<u64> {
    world
        .countries
        .iter()
        .filter(|(_, c)| c.mobilized())
        .map(|(id, _)| id.0)
        .collect()
}

// --------------------------------------------------------------------------- //
// 1 ラウンドの指標レコード (metrics.csv の 1 行)
// --------------------------------------------------------------------------- //

/// ラウンドごとの指標 (metrics.csv の 1 行)．
#[derive(Debug, Clone, Serialize)]
pub struct RoundMetric {
    /// ラウンド (0 始まり)．
    pub round: u64,
    /// 同盟分割 MI (vs 史実) ∈ [0,1]．
    pub alliance_mi: f64,
    /// 宣戦布告対集合の Jaccard (vs 史実) ∈ [0,1]．
    pub declaration_jaccard: f64,
    /// 総動員集合の Jaccard (vs 史実) ∈ [0,1]．
    pub mobilization_jaccard: f64,
    /// 宣戦布告 (W) 対の総数．
    pub n_conflicts: u64,
    /// 総動員国数．
    pub n_mobilized: u64,
    /// 同盟クラスタ数．
    pub n_alliance_clusters: u64,
    /// このラウンドで世界大戦が勃発済みか (1/0)．
    pub war_outbreak: u8,
}

/// 与えられた world から 1 ラウンド分の指標を計算する．
pub fn compute_round_metric(world: &WarWorld, round: u64, war_threshold: usize) -> RoundMetric {
    let sim_part = alliance_partition(world);
    let hist_part = historical_alliance_partition();
    let mi = alliance_mi(&sim_part, &hist_part);

    let sim_wars = war_pair_set(world);
    let hist_wars = historical_war_pairs();
    let decl_j = jaccard(&sim_wars, &hist_wars);

    let sim_mob = mobilized_set(world);
    let hist_mob = historical_mobilized();
    let mob_j = jaccard(&sim_mob, &hist_mob);

    let n_conflicts = sim_wars.len() as u64;
    let outbreak = sim_wars.len() >= war_threshold;

    RoundMetric {
        round,
        alliance_mi: mi,
        declaration_jaccard: decl_j,
        mobilization_jaccard: mob_j,
        n_conflicts,
        n_mobilized: sim_mob.len() as u64,
        n_alliance_clusters: sim_part.len() as u64,
        war_outbreak: if outbreak { 1 } else { 0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{build_countries, Scenario};
    use crate::world::{Country, Profile, Stance, WarWorld};
    use socsim_core::{AgentId, SimClock};

    fn world() -> WarWorld {
        let (countries, boards) = build_countries(Scenario::Wwi, None);
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
    fn jaccard_basic() {
        let a: BTreeSet<u64> = [1, 2, 3].into_iter().collect();
        let b: BTreeSet<u64> = [2, 3, 4].into_iter().collect();
        // inter {2,3}=2, union {1,2,3,4}=4 → 0.5
        assert!((jaccard(&a, &b) - 0.5).abs() < 1e-9);
        let e: BTreeSet<u64> = BTreeSet::new();
        assert!((jaccard(&e, &e) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mi_identical_partition_is_one() {
        let p = historical_alliance_partition();
        assert!((alliance_mi(&p, &p) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mi_in_unit_range() {
        let sim = vec![vec![0u64, 1], vec![2, 3, 4, 5, 6, 7]];
        let mi = alliance_mi(&sim, &historical_alliance_partition());
        assert!((0.0..=1.0).contains(&mi), "mi={mi}");
    }

    #[test]
    fn alliance_partition_uses_transitive_closure() {
        let mut w = world();
        // A-B alliance already set; add B-? to extend cluster.
        // initial WWI has A-B (0-1) and C-D,C-E (2-3,2-4) alliances.
        let part = alliance_partition(&w);
        // {0,1} should be one cluster.
        assert!(part.iter().any(|g| g.contains(&0) && g.contains(&1)));
        // {2,3,4} should be one cluster (C-D, C-E transitive).
        assert!(part
            .iter()
            .any(|g| g.contains(&2) && g.contains(&3) && g.contains(&4)));
        // mutate: declare a war and recompute conflicts.
        w.boards
            .get_mut(&AgentId(0))
            .unwrap()
            .set_relation(AgentId(2), Relation::War);
        let m = compute_round_metric(&w, 0, 3);
        assert!(m.n_conflicts >= 1);
        let _ = Country::new("x", Profile::default(), Stance::Neutral);
    }

    #[test]
    fn alliance_network_matches_partition_clusters() {
        let w = world();
        let net = alliance_network(&w);
        // socsim-net の連結成分数 = union-find クラスタ数 (相互検算)．
        assert_eq!(net.connected_components(), alliance_partition(&w).len());
        assert_eq!(net.node_count(), w.n_countries());
    }

    #[test]
    fn round_metric_ranges() {
        let w = world();
        let m = compute_round_metric(&w, 0, 3);
        assert!((0.0..=1.0).contains(&m.alliance_mi));
        assert!((0.0..=1.0).contains(&m.declaration_jaccard));
        assert!((0.0..=1.0).contains(&m.mobilization_jaccard));
        // no wars at init → no outbreak.
        assert_eq!(m.war_outbreak, 0);
        assert_eq!(m.n_conflicts, 0);
    }
}
