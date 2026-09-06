[English](visualization.md) | [日本語](visualization.ja.md)

# Visualization

The Python package `waragent-tools` (uv workspace member; module `waragent_tools`) reads the Rust outputs and renders figures. Install with `uv sync` at the repository root, then run via `uv run waragent-tools <subcommand>`.

## `visualize`

Reads a run directory's `metrics.csv` (long form) and the `x.hua2024.action` events in `events.jsonl`, and writes one 2×2 figure `war_dynamics.png`:

1. **Alliance / war network** (final round) — a `networkx` circular graph; alliance (M) edges in green, war (W, incl. escalation joins) edges as red dashes; nodes labelled with anonymized country names (Country A..H).
2. **Board transitions** — per-round counts of `declare_war` / `alliance` / `non_aggression` / `mobilize` actions.
3. **Metric time series** — `alliance_mi`, `declaration_jaccard`, `mobilization_jaccard` vs round.
4. **Conflict scale** — `n_conflicts` (war pairs) and `n_mobilized` (mobilized countries) vs round (escalation).

Which run is shown is answered by runvault when `--results-dir` is omitted (`runvault path --experiment waragent --latest --subcommand run --standalone`). Figures go *beside* the run directory (`results/waragent/figures/<run_slug>/`), because `manifest.csv` is settled by `finish()` and anything added afterwards would carry no hash.

```bash
uv run waragent-tools visualize
uv run waragent-tools visualize --results-dir "$(runvault path --experiment waragent --latest --subcommand run --standalone)"
uv run waragent-tools visualize --output-dir out
```

## `visualize-sweep`

Rebuilds the one-row-per-execution sweep table from the sweep parent's child runs (runvault keeps no such table on disk) and writes:

- `sweep_outbreak_heatmap.png` — war-outbreak rate over trigger × stance.
- `sweep_alliance_mi_heatmap.png` — mean `final_alliance_mi` over trigger × stance.
- `sweep_trigger_bars.png` — per-trigger war-outbreak rate vs cold-war rate (the "small trigger still escalates / null stays cold" trend).

```bash
uv run waragent-tools visualize-sweep
uv run waragent-tools visualize-sweep --sweep-dir "$(runvault path --experiment waragent --latest --subcommand sweep)"
```

Note: a legacy `sweep_summary.csv`, if present, is read instead; there the `null` trigger label is read as the literal string (not parsed as a missing value).

## `reproduce`

Reads the `reproduce` parent (its `scope=sweep` alliance-polarization gap) and its per-condition child runs, prints the per-trigger table, and writes:

- `table2_alliance_escalation.png` — per-trigger `alliance_mi` and `n_conflicts` time series (alliance polarization and escalation).
- `table5_trigger_compare.png` — per-trigger cross-comparison bars: war-outbreak / cold-war flags and final alliance MI.

The anchor bands and their PASS/off verdicts are not printed here. The bands are this replication's own, not the paper's claim, so they stay in the Rust binary's console output — putting the same threshold in two places lets them drift apart.

```bash
uv run waragent-tools reproduce
uv run waragent-tools reproduce --run --mock --quick   # run the Rust binary first (offline)
uv run waragent-tools reproduce --json                 # print the summary only
```

## `show-experiment-settings`

Pretty-prints the run's conditions (`config.json` `parameters`), the `llm` block of `run.json`, and the run-scope metrics (call count, cache hits, final round, cold-war flag, escalation round). Legacy `config.json` / `sweep_config.json` / `run_metadata.json` are read as before.

```bash
uv run waragent-tools show-experiment-settings
uv run waragent-tools show-experiment-settings --results-dir "$(runvault path --experiment waragent --latest --subcommand sweep)"
uv run waragent-tools show-experiment-settings --json
```

## Font note

The scripts set `font.family = "Hiragino Sans"` for Japanese labels (macOS). On other platforms, swap in an installed CJK font if labels render as boxes.
