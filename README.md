# Role-Restricted Evidence Validation for Agentic Equity Research

> A deterministic, multi-layer audit solver for reconstructing the complete consistency chain of a Large Language Model–based multi-agent equity research workflow.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reference](https://img.shields.io/badge/arXiv-2508.11152-red.svg)](https://arxiv.org/pdf/2508.11152)

**Reference paper:** Zhao, T., Lyu, J., Jones, S., Garber, H., Pasquali, S., & Mehta, D. (2025). *AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions.* https://arxiv.org/pdf/2508.11152

---

## Abstract

This repository implements a deterministic auditing solver, `multi_agent_equity_consistency_audit`, designed to reconstruct and validate the complete evidentiary, dialectical, behavioural, and econometric audit chain of an agentic equity research workflow patterned after the AlphaAgents framework of Zhao et al. (2025). The benchmark formalises an equity-research pipeline in which three role-restricted specialist agents — a valuation agent constrained to price-volume evidence, a fundamental agent constrained to filing-derived evidence, and a sentiment agent constrained to news-derived evidence — produce risk-profile-conditioned recommendations that are subsequently reconciled through a structured multi-agent debate. The solver verifies four interlocking audit layers: role-access compliance and claim-evidence support; directed-graph reconstruction of debate logs with PageRank-based speaker influence and terminal-consensus reconciliation; realised portfolio performance under an equal-weight, BUY-only construction with rolling Sharpe diagnostics and an alpha-beta excess-return regression; and risk-profile monotonicity together with a constrained convex projection on risk-averse inclusion scores. The solver emits a single structured JSON object that summarises the entire audit and assigns a discrete consistency signature drawn from the set `{fully_consistent, portfolio_consistent_with_evidence_warnings, risk_profile_inconsistent, globally_inconsistent}`.

---

## 1. Background and Motivation

The deployment of multi-agent Large Language Model architectures for portfolio construction, while empirically promising, raises substantive concerns regarding evidentiary integrity, role compartmentalisation, and the reconciliation of dialectical disagreement among specialist agents. The AlphaAgents framework introduced by Zhao et al. (2025) demonstrates that a coordinated debate among role-prompted agents can outperform single-agent baselines on a curated equity universe. However, the validity of such a system rests not solely on realised portfolio performance but on the integrity of the upstream evidence chain. A portfolio that outperforms its benchmark while relying upon claims unsupported by permissible evidence, or upon a debate whose terminal consensus fails to reconcile with the published decision table, constitutes an empirically successful yet methodologically incoherent system. The present solver addresses this gap by treating the multi-agent benchmark as a single coupled audit problem rather than a disjoint collection of validation checks.

The intellectual premise of the audit is that a faithful reconstruction must enforce a strict separation between the analysis window — in which evidence is gathered and decisions are formed — and the portfolio window — in which realised returns are evaluated. Any leakage of portfolio-window information into the decision-formation process would constitute a lookahead violation. The solver therefore enforces, by contractual validation, that the portfolio window begins strictly after the analysis window terminates, and that all evidence timestamps respect this temporal partition.

The full reference paper is available at https://arxiv.org/pdf/2508.11152.

---

## 2. Methodological Architecture

The solver decomposes the audit into four canonical layers, each of which contributes a normalised component score on the unit interval $[0,1]$ to a global consistency index.

### 2.1 Evidence Layer

The evidence layer constructs a canonical mapping from evidence identifiers to evidence objects across four sources: price-volume records, filing chunks, news articles, and structured debate records. Each claim is classified into exactly one of five mutually exclusive statuses:

```
supported | missing_evidence | role_access_violation | decision_mismatch | unsupported
```

A claim attains `supported` status if and only if the referenced evidence exists in the canonical map, the evidence channel is permitted for the agent's role under the feature-schema contract, and the report's final recommendation agrees with the corresponding decision-table record under the matched ticker, agent, and risk profile. Role-access compliance is enforced through the canonical mapping `{valuation → price-volume, fundamental → filings, sentiment → news, multi_agent → reports/debate}`, with any cross-channel reference classified as a `role_access_violation`.

### 2.2 Debate Layer

For each debate identifier, a directed acyclic graph $G = (V, E)$ is constructed using `networkx.DiGraph`, in which each message constitutes a node and each `reply_to` reference induces a directed edge from the referenced message to the replying message. The graph must satisfy three structural invariants: acyclicity, enforced by round-monotonic edge ordering; minimum specialist participation, requiring at least the configured number of turns from each of the three specialist agents; and a unique terminal manager message bearing the configured termination token.

Speaker influence is computed via the standard PageRank algorithm

$$
PR(v) = \frac{1-d}{|V|} + d \sum_{u \in N^{-}(v)} \frac{PR(u)}{|N^{+}(u)|}
$$

with subsequent aggregation by speaker. Unresolved dissent is flagged when the terminal decision diverges from at least two specialists' final explicit decisions, providing a structural diagnostic for premature consensus.

### 2.3 Portfolio Layer

For each `(agent, risk_profile)` pair, an equal-weight portfolio is constructed from BUY decisions only, with HOLD and SELL excluded by the canonical `HOLD_POLICY = "exclude"` contract. Daily simple returns are computed from adjusted close prices, and the standard suite of risk-adjusted performance statistics is reported:

| Metric | Formulation |
|---|---|
| Cumulative return | $C_p = \prod_t (1 + R_{p,t}) - 1$ |
| Annualised volatility | $\sigma_p = \sqrt{252} \cdot \mathrm{std}(R_{p,t})$ |
| Sharpe ratio | $S_p = \dfrac{\mathrm{mean}(R_{p,t} - R_{f,t})}{\mathrm{std}(R_{p,t} - R_{f,t})} \cdot \sqrt{252}$ |
| Maximum drawdown | $D_p = \min_t \left( \dfrac{V_{p,t}}{\max_{\tau \le t} V_{p,\tau}} - 1 \right)$ |
| Rolling Sharpe (window $w$) | $S^{(w)}_{p,t} = \dfrac{\mathrm{mean}(R^{e}_{t-w+1:t})}{\mathrm{std}(R^{e}_{t-w+1:t})} \cdot \sqrt{252}$ |

Sample standard deviations employ the unbiased denominator $(n-1)$. Where the excess-return standard deviation collapses below the configured tolerance, the corresponding Sharpe value is set to `None` rather than infinity, in accordance with numerical hygiene.

An ordinary least squares regression on excess returns,

$$
R_{p,t} - R_{f,t} = \alpha + \beta (R_{b,t} - R_{f,t}) + \epsilon_t,
$$

is fitted via `statsmodels.api.OLS` to characterise each portfolio's relationship to the equal-weight universe benchmark, with $\alpha$ classified as `positive_alpha`, `negative_alpha`, or `flat_alpha` under the tie tolerance.

### 2.4 Risk-Profile Layer and Convex Projection

The risk-profile layer aggregates January-window stock features across selected sets and tests the empirically motivated monotonicity condition that risk-averse selections should exhibit lower mean annualised volatility, lower mean equal-weight beta, and less severe mean drawdown than risk-neutral selections. Pairwise Jaccard similarities quantify selection overlap across profiles.

For each agent, a constrained convex projection is solved via `cvxpy`:

$$
\min_{x} \sum_i (x_i - b_i)^2
$$

subject to

$$
0 \le x_i \le 1, \quad \sum_i x_i \ge 1, \quad \sum_i x_i (v_i - \bar{v}_{\text{neutral}}) \le 0,
$$

where $b_i \in \{0,1\}$ denotes the observed risk-averse BUY indicator, $v_i$ the January annualised volatility, and $\bar{v}_{\text{neutral}}$ the neutral-set mean volatility. Solver fallback proceeds OSQP → SCS, with explicit reporting of the solver invoked and the terminal status.

---

## 3. Numerical Precision Protocol

Given the precision-sensitive character of compound returns, drawdown extrema, Sharpe-ratio components, and rolling-window aggregates, all intermediate arithmetic is conducted in `mpmath` arbitrary-precision floating-point at eighty decimal digits. Native Python `float` and `int` conversions occur exclusively at the boundary of the final JSON construction. Tolerance comparisons employ a configurable `tie_tol` parameter (default $10^{-12}$), which governs ranking ties, alpha sign classification, and component-score equality tests. All emitted floating-point quantities are rounded to twelve decimal places, with negative zero canonicalised to positive zero.

---

## 4. Repository Structure

The repository adopts a flat, functionally partitioned layout in which inputs, deliverables, diagnostics, and the validation harness are segregated into discrete top-level directories. The canonical organisation is reproduced below.

```
multi_agent_equity_consistency_audit/
├── data/                                       # canonical benchmark inputs (14 files)
│   ├── agent_decisions.csv
│   ├── agent_reports.jsonl
│   ├── claim_evidence_links.csv
│   ├── config.json
│   ├── debate_logs.jsonl
│   ├── feature_schema.json
│   ├── filings_chunks.jsonl
│   ├── filings_metadata.csv
│   ├── news_articles.jsonl
│   ├── price_evidence.jsonl
│   ├── prices.csv
│   ├── risk_free.csv
│   ├── stock_features.csv
│   └── universe.csv
├── figures/                                    # diagnostic visualisations
│   ├── 01_global_consistency_score.png
│   ├── 02_multi_agent_risk_neutral_vs_benchmark.png
│   ├── 03_risk_return_scatter.png
│   ├── 04_buy_selection_matrix.png
│   ├── 05_return_correlation_heatmap.png
│   └── 06_debate_message_graph.png
├── results/                                    # canonical audit artefact
│   └── output.json
├── tests/                                      # deterministic validation harness
│   └── test.py
├── run_audit.py                                # top-level invocation entry point
├── .gitignore
└── README.md
```

The principal solver function `multi_agent_equity_consistency_audit` is invoked through the `run_audit.py` driver at the repository root, which orchestrates the binding of canonical input paths under `data/` and the persistence of the final JSON artefact under `results/output.json`. The validation suite under `tests/test.py` is independent of `run_audit.py` and constructs its own ephemeral fixtures via `tempfile.TemporaryDirectory` to ensure hermetic test execution.

---

## 5. Input Data Specification

The benchmark dataset comprises fourteen canonical files structured as a coupled relational system over the canonical axes:

```python
AGENTS            = ["valuation", "fundamental", "sentiment", "multi_agent"]
SPECIALIST_AGENTS = ["valuation", "fundamental", "sentiment"]
RISK_PROFILES     = ["risk_neutral", "risk_averse", "risk_seeking"]
DECISIONS         = ["BUY", "SELL", "HOLD"]
```

The configuration file `config.json` enforces strict adherence to the fixed benchmark contract: `TRADING_DAYS_PER_YEAR = 252`, `ROLLING_SHARPE_WINDOW = 20`, `HOLD_POLICY = "exclude"`, and `PORTFOLIO_WEIGHTING = "equal_weight"`. Any deviation from these conventions, or contradiction with the canonical axis ordering, elicits a `ValueError` accompanied by a non-empty diagnostic message.

The feature schema `feature_schema.json` codifies role-restricted evidence access: the valuation agent may consult only ticker identity, OHLC fields, adjusted close, and volume; the fundamental agent only ticker identity and filing-derived chunks; the sentiment agent only ticker identity and news fields; and the multi-agent layer only specialist reports, debate evidence, and consensus decisions. The remaining twelve files partition into three categories. The first comprises tabular reference data (`universe.csv`, `prices.csv`, `risk_free.csv`, `stock_features.csv`, `filings_metadata.csv`, `agent_decisions.csv`, `claim_evidence_links.csv`). The second comprises JSON-Lines evidence corpora (`price_evidence.jsonl`, `filings_chunks.jsonl`, `news_articles.jsonl`). The third comprises JSON-Lines agentic artefacts (`agent_reports.jsonl`, `debate_logs.jsonl`).

---

## 6. Output Specification

The solver returns and persists a single JSON object, written by default to `results/output.json`, with six top-level sections.

| Section | Content |
|---|---|
| `dataset_profile` | Cardinality summary across tickers, agents, decisions, reports, claims, debates, observations |
| `evidence_audit` | Claim status counts, agent-level support rates, claim-level results |
| `debate_audit` | Per-debate graph metrics, terminal validity, consensus reconciliation, speaker PageRank |
| `portfolio_audit` | Per-portfolio metrics, benchmark metrics, deterministic rankings |
| `risk_profile_audit` | Selected sets, Jaccard similarities, feature means, monotonicity violations, projection results |
| `summary` | Component-wise booleans, global consistency score, audit signature |

The global consistency score is defined as the unweighted arithmetic mean of five component scores: the evidence support fraction, the debate validity fraction, the portfolio finiteness fraction, the risk-profile compliance fraction, and the convex-projection feasibility fraction. The audit signature is assigned via the priority rule:

| Signature | Code | Condition |
|---|---|---|
| `fully_consistent` | 3 | All five component scores equal to one within `tie_tol` |
| `portfolio_consistent_with_evidence_warnings` | 2 | Portfolio and debate scores at unity; evidence or risk-profile below unity |
| `risk_profile_inconsistent` | 1 | At least one monotonicity violation; all consensus reconciled |
| `globally_inconsistent` | 0 | Otherwise |

---

## 7. Installation and Invocation

### 7.1 Dependencies

The solver depends solely upon the Python standard library and the following third-party packages:

```bash
pip install pandas numpy mpmath statsmodels networkx cvxpy
```

### 7.2 Reference Invocation

The repository may be executed end-to-end through the top-level driver:

```bash
git clone https://github.com/himadri027roy/multi_agent_equity_consistency_audit.git
cd multi_agent_equity_consistency_audit
python run_audit.py
```

Programmatic invocation, intended for embedding within a downstream evaluation pipeline, proceeds as follows:

```python
from run_audit import multi_agent_equity_consistency_audit

result = multi_agent_equity_consistency_audit(
    universe_path="data/universe.csv",
    config_path="data/config.json",
    feature_schema_path="data/feature_schema.json",
    prices_path="data/prices.csv",
    risk_free_path="data/risk_free.csv",
    stock_features_path="data/stock_features.csv",
    price_evidence_path="data/price_evidence.jsonl",
    filings_metadata_path="data/filings_metadata.csv",
    filings_chunks_path="data/filings_chunks.jsonl",
    news_articles_path="data/news_articles.jsonl",
    agent_reports_path="data/agent_reports.jsonl",
    claim_evidence_links_path="data/claim_evidence_links.csv",
    debate_logs_path="data/debate_logs.jsonl",
    agent_decisions_path="data/agent_decisions.csv",
    output_path="results/output.json",
    tie_tol=1e-12,
)

print(result["summary"]["audit_signature"])
print(result["summary"]["global_consistency_score"])
```

---

## 8. Validation Suite

The validation harness, located at `tests/test.py`, comprises forty deterministic test cases organised along the following thematic axes.

- **Schema and contract validation** (Cases 10–15, 29–30): malformed JSON/JSONL/CSV, missing keys, duplicate identifiers, ticker-coverage inconsistencies, contract violations.
- **Evidence-layer correctness** (Cases 1–2, 16–17, 28, 34, 40): claim status classification, role-access enforcement, decision-mismatch detection, missing-evidence handling.
- **Debate-layer correctness** (Cases 3, 18–21, 39): graph reconstruction, PageRank aggregation, terminal validity, unresolved dissent, duplicate manager rejection.
- **Portfolio-layer correctness** (Cases 4–5, 9, 22–23, 27, 31–33, 36, 38): metric reproduction, risk-free fill semantics, rolling Sharpe non-triviality, OLS residual degrees of freedom, drawdown ranking by magnitude, configurable risk-free series naming.
- **Risk-profile correctness** (Cases 6, 24, 35, 37): selected sets, Jaccard similarities, monotonicity violation enumeration, signature assignment.
- **Determinism and rounding** (Cases 8, 25): whitespace invariance, twelve-decimal rounding.
- **Library invocation** (Case 26): verification of `statsmodels.OLS`, `networkx.DiGraph`, and `cvxpy` engagement.

The full suite is executed via:

```bash
python tests/test.py
```

A successful execution is expected to terminate with `PASS=40, FAIL=0, ERROR=0`.

---

## 9. Diagnostic Visualisations

The `figures/` directory contains six diagnostic visualisations generated from a representative benchmark run.

1. **`01_global_consistency_score.png`** — bar chart of the scalar consistency index annotated with the assigned signature.
2. **`02_multi_agent_risk_neutral_vs_benchmark.png`** — time-series comparison of cumulative wealth between the multi-agent risk-neutral portfolio and the equal-weight benchmark over the portfolio window.
3. **`03_risk_return_scatter.png`** — annualised volatility against cumulative return across all twelve `(agent, risk_profile)` portfolios.
4. **`04_buy_selection_matrix.png`** — binary heatmap of selection inclusion across the universe by portfolio.
5. **`05_return_correlation_heatmap.png`** — pairwise correlation matrix of ticker-level daily returns over the portfolio window.
6. **`06_debate_message_graph.png`** — node-link visualisation of a representative debate's directed message graph.

---

## 10. Theoretical Significance

This artefact contributes to the emerging literature at the intersection of agentic finance, multi-agent reasoning, and reproducible evaluation by establishing a deterministic, schema-driven audit protocol that decouples the validation of agentic process integrity from the validation of realised economic performance. A multi-agent system that produces ex-post outperformance while violating evidentiary access constraints, debate-graph structure, or risk-profile monotonicity is, under the proposed framework, classified as `globally_inconsistent` notwithstanding its return profile. This separation of concerns provides a rigorous foundation for the comparative evaluation of agentic equity research systems and establishes evaluative criteria that extend beyond the conventional Sharpe-and-cumulative-return paradigm.

---

## 11. References

1. Zhao, T., Lyu, J., Jones, S., Garber, H., Pasquali, S., & Mehta, D. (2025). *AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions.* arXiv preprint arXiv:2508.11152. https://arxiv.org/pdf/2508.11152
2. Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search engine. *Computer Networks and ISDN Systems*, 30(1–7), 107–117.
3. Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business*, 39(1), 119–138.
4. Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. *Mathematical Programming Computation*, 12(4), 637–672.
5. O'Donoghue, B., Chu, E., Parikh, N., & Boyd, S. (2016). Conic optimization via operator splitting and homogeneous self-dual embedding. *Journal of Optimization Theory and Applications*, 169(3), 1042–1068.
6. Diamond, S., & Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex optimization. *Journal of Machine Learning Research*, 17(83), 1–5.
7. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In *Proceedings of the 9th Python in Science Conference*.
8. Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference*.
