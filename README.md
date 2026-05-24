**English** | [日本語](README.ja.md)

# War and Peace (WarAgent): LLM-based Multi-Agent Simulation of World Wars — Hua et al. (2024)

A reimplementation of the LLM-driven country-agent diplomacy model of Hua et al. (2024), "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars" (arXiv:2311.17227). A small set of **country agents** (WWI = 8) reason over rounds about alliances, war declarations, treaties and mobilization in an international crisis. Each country has a six-dimension profile (Leadership / Military / Resources / Historical Background / Key Policy / Public Morale) and chooses one action per round from a seven-action space (Wait / Mobilize / Declare War / Alliance / Non-aggression / Peace / Message). The huge multi-agent context is compressed by a **Board** (international relations: W/M/T/P) and **Stick** (domestic state: mobilization) translated into a short context paragraph, and country names are **anonymized** (Country A..H) so the LLM reasons rather than recites history. The deterministic [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) core handles board/relation resolution, escalation and all metrics, while the non-deterministic LLM layer is confined to one decision mechanism and pseudo-determinised via the `socsim-llm` crate (prompt→response cache + `temperature=0` + fixed seed).

## Two-layer determinism (read this first)

LLM output is **outside** socsim's bit-reproducibility. The design therefore splits into two layers:

- **Deterministic socsim core** — scenario/board initialisation, activation order, publicity-based propagation, alliance/war/treaty resolution, escalation (allies joining a war), board & mobilization updates, and all metrics (alliance mutual information, declaration/mobilization Jaccard, war-outbreak, escalation round). Given a seed this reproduces bit-for-bit (`ctx.rng`, ChaCha20 `SimRng`).
- **Non-deterministic LLM layer** — the single `Decision` mechanism (`CountryDecisionMechanism`): each country runs a four-step guided reasoning (identify allies → recognize adversaries → recommend action → final action) plus a configurable secretary verification pass. Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed seed. The provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses, so a rerun is free and stable. **LLM calls per round = n_countries × (1 + secretary_passes)**; `--secretary-passes` defaults to `1` to bound the call budget. Each run writes `run_metadata.json` recording the model, endpoint, temperature, seed and cache-hit rate. Because the local default model (`llama3.2`) differs from the paper's `GPT-4`/`Claude-2`, reproduction fidelity is **moderate (△–○)**: target the *trends* (war tends to break out; a null trigger stays in a cold-war state; alliance MI is above random) rather than the exact Table-2 percentages.

> This project standardises on the `socsim-llm` crate for the LLM layer; it does **not** use `reqwest` or `sha2` (socsim-llm owns the HTTP transport and the prompt-cache hashing), superseding design §4.2/§7 and matching li2024 / zhao2024 / ren2024 / gao2023.

## Relationship structure

Countries are few, so the per-country **Board** relations (`(owner, other) → W/M/T/P`) are kept as an explicit `BTreeMap` matrix and are the **source of truth** (partial-information per-country boards). A global undirected alliance graph is additionally derived as a `socsim-net::SocialNetwork` for cluster extraction / visualization; the union-find partition over the boards and the network's connected-component count agree (a mutual check). There is no spatial grid, so `socsim-grid` is not a dependency.

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-net + socsim-llm with the Ollama+OpenAI backends)
cargo build --release

# Make sure a local Ollama is running and a model is pulled, e.g.:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# Optional OpenAI fallback:
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini

# A small smoke run (4 countries, 2 rounds) — cheap to verify the live path:
cargo run --release -- run --scenario wwi-small --rounds 2 --runs 1 --secretary-passes 1 --seed 42

# The paper-scale WWI base experiment (8 countries):
#   cargo run --release -- run --scenario wwi --trigger archduke-assassination --rounds 6 --runs 7 --seed 42

# Install the Python visualization tools (at the workspace root)
uv sync

# Visualize the most recent run (alliance network, board transitions, metric time series)
uv run waragent-tools visualize

# Inspect the run's settings and LLM metadata
uv run waragent-tools show-experiment-settings --results-dir results/latest
```

### Offline (no-LLM) smoke

The full round loop, output writers and Python visualization can be exercised without any live LLM via a scripted mock client:

```bash
cargo run --release --example mock_smoke -- results
uv run waragent-tools visualize
```

### Sensitivity sweep (trigger × stance)

```bash
cargo run --release -- sweep \
    --scenario wwi \
    --trigger-values null,naval-incident,dardanelles \
    --stance-values conservative,aggressive \
    --rounds 6 --runs 7 --seed 42

uv run waragent-tools visualize-sweep
```

## Outputs

Each `run` writes `results/{timestamp}/` (with `results/latest` symlink):

| File | Contents |
|---|---|
| `config.json` | the run configuration |
| `metrics.csv` | per-round metrics (`alliance_mi`, `declaration_jaccard`, `mobilization_jaccard`, `n_conflicts`, `n_mobilized`, `n_alliance_clusters`, `war_outbreak`) |
| `events.csv` | action log (`round, actor, action, target, publicity`) |
| `run_metadata.json` | LLM model / endpoint / temperature / seed / cache-hit rate + macro outcomes |

A `sweep` writes `results/{timestamp}_sweep/` with `sweep_config.json` and `sweep_summary.csv`.

## Documentation

- [Architecture](docs/architecture.md) ([日本語](docs/architecture.ja.md))
- [CLI reference](docs/cli.md) ([日本語](docs/cli.ja.md))
- [Visualization](docs/visualization.md) ([日本語](docs/visualization.ja.md))

## Scope

- **Phase 1** (core model: country agents + boards + Board/Stick + event log; five mechanisms over six phases; LLM decision layer + cache; WWI base run) — done.
- **Phase 2** (`sweep` over trigger × stance; `visualize` / `visualize-sweep` / `show-experiment-settings`) — done.
- **Phase 3** (`reproduce` for Table 2–5 batch + counterfactual analysis, WWII / Warring-States scenarios, de-anonymization) — deferred. Extension points are left in place: the `Scenario` enum, configurable `secretary_passes`, and the documented de-anonymization map (A=Germany, B=Austria-Hungary, …) in `config.rs`.

## Reference

Hua, W., Fan, L., Li, L., Mei, K., Ji, J., Ge, Y., Hemphill, L., & Zhang, Y. (2023/2024). War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars. *arXiv preprint* arXiv:2311.17227.

## License

MIT — see [LICENSE](LICENSE).

---
*This file was generated by Claude Code.*
