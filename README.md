# Multi-Agent Equity Consistency Audit

A deterministic Python project for auditing multi-agent equity research systems. The project checks whether specialist agents use the correct evidence, whether their reports are consistent with the decision table, whether debate-based consensus is valid, and whether the resulting portfolios behave sensibly across different risk profiles.

---

## 1. Project Overview

This project studies a multi-agent equity research workflow where different agents analyze the same stock universe from different perspectives.

The system contains four types of agents:

- **Valuation agent**: focuses on price, volume, and market-based evidence.
- **Fundamental agent**: focuses on filing-derived financial evidence.
- **Sentiment agent**: focuses on news and textual sentiment evidence.
- **Multi-agent system**: combines the specialist agents through debate and produces a final consensus recommendation.

The main goal is to audit the full decision chain:

```text
Evidence → Agent Report → Claim Support → Debate → Consensus Decision → Portfolio Performance → Risk-Profile Consistency
