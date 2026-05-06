# Multi-Agent Equity Consistency Audit

A deterministic Python audit framework for evaluating multi-agent equity research systems across evidence validation, debate consistency, risk-profile behavior, and realized portfolio performance.

---

## 1. Project Overview

This project implements a reproducible audit system for a multi-agent equity-research workflow. The benchmark assumes that several specialist agents analyze stocks under different information constraints and then produce investment recommendations through a debate-and-consensus process.

The system audits four types of agents:

- **Valuation Agent** — uses price, volume, and market-based evidence.
- **Fundamental Agent** — uses filing-derived financial evidence.
- **Sentiment Agent** — uses news-title and news-body evidence.
- **Multi-Agent System** — combines specialist reports and debate records to generate a final consensus decision.

The core objective is not simply to backtest portfolios. The objective is to verify whether the entire decision chain is internally consistent:

```text
Evidence → Agent Report → Claim Support → Debate → Consensus Decision → Portfolio Performance → Risk-Profile Consistency
