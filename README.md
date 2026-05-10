# Role-Restricted Evidence Validation for Agentic Equity Research

> A deterministic, multi-layer audit solver for reconstructing the complete consistency chain of a Large Language Model–based multi-agent equity research workflow.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reference](https://img.shields.io/badge/arXiv-2508.11152-red.svg)](https://arxiv.org/pdf/2508.11152)

**Reference paper:** Zhao, T., Lyu, J., Jones, S., Garber, H., Pasquali, S., & Mehta, D. (2025). *AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions.* 



---
## Why this project matters for quant / model risk roles

This project is relevant to quantitative finance, AI model risk, and model validation because it audits the full decision chain of an LLM-based equity research workflow rather than only evaluating final portfolio returns.

- Implements a full audit pipeline from raw evidence, agent reports, and debate logs to realized portfolio performance.

- Enforces role-restricted evidence use across valuation, fundamental, sentiment, and multi-agent roles, flagging unsupported claims, missing evidence, decision mismatches, and role-access violations.

- Reconstructs multi-agent debate graphs, validates terminal consensus, detects unresolved dissent, and measures specialist influence using PageRank.

- Prevents lookahead bias by separating the analysis window from the portfolio evaluation window.

- Builds equal-weight BUY-only portfolios and computes cumulative return, annualized volatility, Sharpe ratio, maximum drawdown, rolling Sharpe, and alpha/beta versus a benchmark.

- Tests whether risk-averse, risk-neutral, and risk-seeking recommendations behave consistently with their stated risk profiles.

- Uses convex optimization to project risk-averse portfolio selections onto a volatility-consistent constraint and reports solver feasibility.

- Applies strict schema validation, deterministic execution, numerical tolerance controls, and reproducible output generation.

- Mirrors bank-style model-risk governance by combining evidence validation, process validation, outcome analysis, monitoring flags, and documented limitations.

---

## Abstract

This repository implements a deterministic auditing solver, `multi_agent_equity_consistency_audit`, for reconstructing and validating the complete consistency chain of an agentic equity research benchmark. The benchmark formalizes a workflow in which three role-restricted specialist agents, a valuation agent constrained to price-volume evidence, a fundamental agent constrained to filing-derived evidence, and a sentiment agent constrained to news-derived evidence, generate risk-profile-conditioned recommendations that are reconciled through structured multi-agent debate. The solver verifies four interlocking audit layers: role-access compliance and claim-evidence support; directed-graph reconstruction of debate logs with speaker-level PageRank aggregation and terminal-consensus reconciliation; realized portfolio performance under equal-weight BUY-only construction with rolling Sharpe diagnostics and alpha-beta excess-return regression; and risk-profile monotonicity with a constrained convex projection of risk-averse inclusion scores. The solver emits a single structured JSON object that summarizes the audit and assigns one discrete consistency signature from `{fully_consistent, portfolio_consistent_with_evidence_warnings, risk_profile_inconsistent, globally_inconsistent}`.

---

## 1. Background and Motivation




The deployment of multi-agent language-model architectures for portfolio construction raises substantive concerns regarding evidentiary integrity, role compartmentalization, and the reconciliation of disagreement among specialist agents. A coordinated debate among role-prompted agents may improve portfolio selection, but realized portfolio performance alone is not sufficient to establish methodological coherence. A portfolio that outperforms its benchmark while relying on unsupported claims, unauthorized evidence channels, or a terminal consensus that fails to reconcile with the dataset decision table is economically successful but audit-inconsistent. This solver treats the benchmark as a single coupled consistency problem rather than a collection of independent validation checks.


The audit enforces a strict separation between the analysis window, where evidence and stock-level features are used to form decisions, and the portfolio window, where realized returns are evaluated. Portfolio-window prices must not be used to form, revise, or reinterpret analysis-window decisions. Available evidence dates are interpreted within the analysis window, while portfolio-window prices are reserved for realized-performance evaluation.

The full reference paper is available at https://arxiv.org/pdf/2508.11152.






---

## 2. Methodological Architecture

The solver decomposes the audit into four canonical layers, each of which contributes a normalised component score on the unit interval $[0,1]$ to a global consistency index.

### 2.1 Evidence Layer

The evidence layer constructs a canonical mapping from evidence identifiers to evidence objects across four sources: price-volume records, filing chunks, news articles, and structured debate records. Each claim is classified into exactly one of five mutually exclusive statuses:

```
supported | missing_evidence | role_access_violation | decision_mismatch | unsupported
```




A claim attains `supported` status if and only if the referenced evidence exists in the canonical map, the evidence channel is permitted for the agent's role under the feature-schema contract, and the report's final recommendation agrees with the corresponding decision-table record under the matched ticker, agent, and risk profile. Role-access compliance is enforced through the canonical mapping `{valuation → price-volume, fundamental → filings, sentiment → news, multi_agent → specialist reports, debate evidence, and consensus decisions}`, with unauthorized cross-channel references classified as `role_access_violation`.

### 2.2 Debate Layer

For each debate identifier, a directed acyclic graph $G = (V, E)$ is constructed using `networkx.DiGraph`, in which each message constitutes a node and each `reply_to` reference induces a directed edge from the referenced message to the replying message. The graph must satisfy three structural invariants: acyclicity, enforced by round-monotonic edge ordering; minimum specialist participation, requiring at least the configured number of turns from each of the three specialist agents; and a unique terminal manager message bearing the configured termination token.




Speaker influence is computed via the standard PageRank algorithm

$$
PR(v) = \frac{1-d}{|V|} + d \sum_{u \in N^{-}(v)} \frac{PR(u)}{|N^{+}(u)|}
$$

with subsequent aggregation by speaker. Unresolved dissent is flagged when the terminal decision diverges from at least two specialists' final explicit decisions, providing a structural diagnostic for premature consensus.

### 2.3 Portfolio Layer
 
For each `(agent, risk_profile)` pair, an equal-weight portfolio is constructed from BUY decisions only, with HOLD and SELL excluded by the canonical `HOLD_POLICY = "exclude"` contract. Daily simple returns are computed from adjusted close prices, and the standard suite of risk-adjusted performance statistics is reported in the formulations that follow.
 
The cumulative return over the portfolio window is given by
 
$$
C_p = \prod_t (1 + R_{p,t}) - 1.
$$
 
The annualised volatility is computed from the sample standard deviation of daily returns,
 
$$
\sigma_p = \sqrt{252} \cdot \mathrm{std}(R_{p,t}).
$$
 
The Sharpe ratio is constructed from the excess return relative to the daily risk-free rate $R_{f,t}$,
 
$$
S_p = \frac{\mathrm{mean}(R_{p,t} - R_{f,t})}{\mathrm{std}(R_{p,t} - R_{f,t})} \cdot \sqrt{252}.
$$
 
The maximum drawdown is the minimum trough-to-peak ratio of the cumulative wealth process 

$$ 
V_{p,t} = \prod_{\tau \le t}(1 + R_{p,\tau}),
$$
 
$$
D_p = \min_t \left( \frac{V_{p,t}}{\max_{\tau \le t} V_{p,\tau}} - 1 \right).
$$
 
The rolling Sharpe ratio over a window of length $w$, computed from the excess-return process 
$$ 
R^{e}_t = R_{p,t} - R_{f,t},
$$

is given by
 
$$
S^{(w)}_{p,t} = \frac{\mathrm{mean}\!\left(R^{e}_{t-w+1:t}\right)}{\mathrm{std}\!\left(R^{e}_{t-w+1:t}\right)} \cdot \sqrt{252}.
$$
 
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

Precision-sensitive intermediate arithmetic is handled with `mpmath`, including tolerance comparisons, return compounding, cumulative-return products, Sharpe-ratio components, rolling-window statistics, drawdown extrema, Jaccard ratios, ranking comparisons, and convex-projection summaries. The configurable tolerance parameter `tie_tol` defaults to `1e-12` and governs equality checks, rank ties, alpha sign classification, zero-volatility Sharpe handling, and audit-signature component comparisons. Native Python numeric types are used only when constructing the final JSON-compatible output. All derived floating-point quantities in the emitted result are rounded to twelve decimal places.



---

## 4. Repository Structure

The repository adopts a flat, functionally partitioned layout in which inputs, deliverables, diagnostics, and the validation harness are segregated into discrete top-level directories. The canonical organisation is reproduced below.

```
multi_agent_equity_consistency_audit/
├── data/
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
├── figures/
│   ├── 01_global_consistency_score.png
│   ├── 02_multi_agent_risk_neutral_vs_benchmark.png
│   ├── 03_risk_return_scatter.png
│   ├── 04_buy_selection_matrix.png
│   ├── 05_return_correlation_heatmap.png
│   └── 06_debate_message_graph.png
├── notebooks/
│   └── multi_agent_equity_consistency_audit.ipynb
├── results/
│   └── output.json
├── src/
│   └── multi_agent_equity_consistency_audit.py
├── tests/
│   └── test.py
├── run_audit.py
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

- schema and contract validation: malformed files, missing keys, duplicate identifiers, invalid labels, inconsistent ticker coverage, and configuration-contract violations;
- evidence-layer correctness: claim status classification, role-access enforcement, decision-mismatch detection, and missing-evidence handling;
- debate-layer correctness: graph reconstruction, terminal-message validation, specialist-turn coverage, consensus reconciliation, and unresolved-dissent detection;
- portfolio-layer correctness: adjusted-close return reconstruction, risk-free-rate filling, cumulative return, volatility, drawdown, Sharpe ratio, rolling Sharpe ratio, and alpha-beta diagnostics;
- risk-profile correctness: selected-set construction, Jaccard similarity, feature aggregation, monotonicity violation enumeration, and convex-projection reporting;
- determinism and rounding: whitespace normalization, stable ranking, tolerance-controlled comparisons, and twelve-decimal output rounding.

The test suite should be executable without internet access and should use only local fixture files.


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



1. Zhao, T., Lyu, J., Jones, S., Garber, H., Pasquali, S., & Mehta, D. (2025). [*AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions.*](https://arxiv.org/pdf/2508.11152)

2. Brin, S., & Page, L. (1998). [The anatomy of a large-scale hypertextual Web search engine.](https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf) *Computer Networks and ISDN Systems*, 30(1–7), 107–117.

3. Sharpe, W. F. (1966). [Mutual fund performance.](https://finance.martinsewell.com/fund-performance/Sharpe1966.pdf) *The Journal of Business*, 39(1), 119–138.

4. Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). [OSQP: An operator splitting solver for quadratic programs.](https://arxiv.org/pdf/1711.08013) *Mathematical Programming Computation*, 12(4), 637–672.

