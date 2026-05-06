import copy
import csv
import json
import math
import tempfile
from datetime import date, timedelta
from pathlib import Path

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm





AGENTS = ["valuation", "fundamental", "sentiment", "multi_agent"]
SPECIALIST_AGENTS = ["valuation", "fundamental", "sentiment"]
RISK_PROFILES = ["risk_neutral", "risk_averse", "risk_seeking"]
DECISIONS = ["BUY", "SELL", "HOLD"]
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]

TRADING_DAYS_PER_YEAR = 252
ROLLING_SHARPE_WINDOW = 20

EXPECTED_TOP_LEVEL_KEYS = [
    "dataset_profile",
    "evidence_audit",
    "debate_audit",
    "portfolio_audit",
    "risk_profile_audit",
    "summary",
]

COMPANY = {
    "AAA": "Alpha Analytics Corp.",
    "BBB": "Beta Balance Inc.",
    "CCC": "Core Cloud Ltd.",
    "DDD": "Delta Dynamics PLC",
    "EEE": "Echo Energy Co.",
}

FEATURES = {
    "AAA": {
        "january_return": 0.040,
        "january_annualized_volatility": 0.100,
        "january_max_drawdown": -0.020,
        "january_beta_to_equal_weight_universe": 0.60,
        "january_average_volume": 10000000,
    },
    "BBB": {
        "january_return": 0.030,
        "january_annualized_volatility": 0.150,
        "january_max_drawdown": -0.030,
        "january_beta_to_equal_weight_universe": 0.80,
        "january_average_volume": 8000000,
    },
    "CCC": {
        "january_return": 0.060,
        "january_annualized_volatility": 0.250,
        "january_max_drawdown": -0.080,
        "january_beta_to_equal_weight_universe": 1.10,
        "january_average_volume": 12000000,
    },
    "DDD": {
        "january_return": -0.020,
        "january_annualized_volatility": 0.350,
        "january_max_drawdown": -0.120,
        "january_beta_to_equal_weight_universe": 1.30,
        "january_average_volume": 6000000,
    },
    "EEE": {
        "january_return": 0.010,
        "january_annualized_volatility": 0.200,
        "january_max_drawdown": -0.060,
        "january_beta_to_equal_weight_universe": 0.95,
        "january_average_volume": 9000000,
    },
}

DECISION_SETS = {
    "valuation": {
        "risk_neutral": {"BUY": {"AAA", "BBB", "CCC"}, "HOLD": {"EEE"}},
        "risk_averse": {"BUY": {"AAA", "BBB"}, "HOLD": {"EEE"}},
        "risk_seeking": {"BUY": {"CCC", "DDD"}, "HOLD": {"AAA"}},
    },
    "fundamental": {
        "risk_neutral": {"BUY": {"AAA", "CCC", "EEE"}, "HOLD": {"BBB"}},
        "risk_averse": {"BUY": {"AAA", "EEE"}, "HOLD": {"BBB"}},
        "risk_seeking": {"BUY": {"CCC", "DDD", "EEE"}, "HOLD": {"AAA"}},
    },
    "sentiment": {
        "risk_neutral": {"BUY": {"BBB", "CCC", "EEE"}, "HOLD": {"AAA"}},
        "risk_averse": {"BUY": {"BBB", "EEE"}, "HOLD": {"AAA"}},
        "risk_seeking": {"BUY": {"CCC", "DDD"}, "HOLD": {"EEE"}},
    },
    "multi_agent": {
        "risk_neutral": {"BUY": {"AAA", "BBB", "CCC"}, "HOLD": {"EEE"}},
        "risk_averse": {"BUY": {"AAA", "BBB"}, "HOLD": {"EEE"}},
        "risk_seeking": {"BUY": {"CCC", "DDD", "EEE"}, "HOLD": {"AAA"}},
    },
}


def _decision_for(agent, risk_profile, ticker):
    spec = DECISION_SETS[agent][risk_profile]
    if ticker in spec.get("BUY", set()):
        return "BUY"
    if ticker in spec.get("HOLD", set()):
        return "HOLD"
    return "SELL"


def _business_dates(start, end):
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    out = []

    while current <= stop:
        if current.weekday() < 5:
            out.append(current.isoformat())
        current += timedelta(days=1)

    return out


def _round12(value):
    if value is None:
        return None
    out = round(float(value), 12)
    return 0.0 if out == -0.0 else out


def _assert_close(observed, expected, tol=1e-11):
    if expected is None or observed is None:
        assert observed is expected
    elif isinstance(expected, float):
        assert isinstance(observed, float), f"Expected float, got {type(observed)!r}"
        assert abs(observed - expected) <= tol, f"{observed!r} != {expected!r}"
    else:
        assert observed == expected


def _assert_nested_close(observed, expected, tol=1e-11):
    if isinstance(expected, dict):
        assert set(observed.keys()) == set(expected.keys())
        for key in expected:
            _assert_nested_close(observed[key], expected[key], tol=tol)
    elif isinstance(expected, list):
        assert observed == expected
    else:
        _assert_close(observed, expected, tol=tol)


def _assert_value_error_nonempty(fn, **kwargs):
    try:
        fn(**kwargs)
        assert False, "Expected ValueError, but no exception was raised."
    except ValueError as exc:
        assert str(exc).strip() != "", "ValueError message should be non-empty."
    except Exception as exc:
        assert False, (
            f"Expected ValueError, but got {type(exc).__name__}: {exc}. "
            "Input-contract violations must raise ValueError."
        )


def _call_valid(fn, **kwargs):
    try:
        return fn(**kwargs)
    except Exception as exc:
        assert False, f"Valid payload unexpectedly raised {type(exc).__name__}: {exc}"


def _assert_json_native(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            assert type(key) is str, f"{path} has non-string key."
            _assert_json_native(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_json_native(child, f"{path}[{index}]")
    else:
        assert value is None or type(value) in (str, int, float, bool), (
            f"{path} has non-JSON-native type {type(value)!r}."
        )
        if isinstance(value, float):
            assert math.isfinite(value), f"{path} has non-finite float."


def _make_valid_dataset():
    analysis_dates = _business_dates("2024-01-02", "2024-01-31")
    portfolio_dates = _business_dates("2024-02-01", "2024-03-15")
    all_dates = analysis_dates + portfolio_dates

    base_prices = {
        "AAA": 100.0,
        "BBB": 80.0,
        "CCC": 60.0,
        "DDD": 50.0,
        "EEE": 40.0,
    }

    drifts = {
        "AAA": 0.0010,
        "BBB": 0.0006,
        "CCC": 0.0014,
        "DDD": -0.0002,
        "EEE": 0.0003,
    }

    waves = {
        "AAA": 0.0030,
        "BBB": -0.0020,
        "CCC": 0.0040,
        "DDD": -0.0035,
        "EEE": 0.0025,
    }

    prices = []

    for ticker_index, ticker in enumerate(TICKERS):
        value = base_prices[ticker]

        for day_index, day in enumerate(all_dates):
            if day_index > 0:
                value *= (
                    1.0
                    + drifts[ticker]
                    + waves[ticker] * math.sin((day_index + 1) * (ticker_index + 2) / 3.0)
                )

            prices.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": f"{value * (1 - 0.001 * (ticker_index + 1)):.12f}",
                    "high": f"{value * (1.004 + 0.0002 * ticker_index):.12f}",
                    "low": f"{value * (0.996 - 0.0001 * ticker_index):.12f}",
                    "close": f"{value * (1.0 + 0.0003 * math.cos(day_index + ticker_index)):.12f}",
                    "adjusted_close": f"{value:.12f}",
                    "volume": str(1000000 + 50000 * ticker_index + 1000 * day_index),
                }
            )

    data = {
        "config": {
            "dataset_type": "unit_test_surrogate_for_alphaagents_benchmark",
            "paper": "AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions",
            "paper_arxiv": "2508.11152",
            "analysis_start": "2024-01-01",
            "analysis_end": "2024-01-31",
            "portfolio_start": "2024-02-01",
            "portfolio_end": "2024-03-15",
            "trading_days_per_year": 252,
            "rolling_sharpe_window": 20,
            "risk_free_series": "DGS1MO",
            "portfolio_weighting": "equal_weight",
            "allowed_agents": list(AGENTS),
            "risk_profiles": list(RISK_PROFILES),
            "allowed_decisions": list(DECISIONS),
            "hold_policy": "exclude",
            "minimum_turns_per_agent": 2,
            "termination_token": "TERMINATE",
        },
        "feature_schema": {
            "valuation": [
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ],
            "fundamental": [
                "ticker",
                "10K",
                "10Q",
                "sec_xbrl_companyfacts",
            ],
            "sentiment": [
                "ticker",
                "news_body",
                "news_title",
                "news_url",
                "news_date",
            ],
            "multi_agent": [
                "valuation_report",
                "fundamental_report",
                "sentiment_report",
                "debate_log",
                "consensus_decision",
            ],
        },
        "universe": [],
        "prices": prices,
        "risk_free": [],
        "stock_features": [],
        "filings_metadata": [],
        "price_evidence": [],
        "filings_chunks": [],
        "news_articles": [],
        "agent_decisions": [],
        "agent_reports": [],
        "claim_evidence_links": [],
        "debate_logs": [],
    }

    for ticker_index, ticker in enumerate(TICKERS):
        feature = FEATURES[ticker]

        data["universe"].append(
            {
                "ticker": ticker,
                "company_name": COMPANY[ticker],
                "sector": "Technology" if ticker in {"AAA", "CCC"} else "Industrial",
                "industry": "Benchmark Test Industry",
                "benchmark_weight": f"{1.0 / len(TICKERS):.12f}",
            }
        )

        data["stock_features"].append(
            {
                "ticker": ticker,
                **{key: str(value) for key, value in feature.items()},
            }
        )

        data["filings_metadata"].append(
            {
                "ticker": ticker,
                "company_name": COMPANY[ticker],
                "cik": str(1000000 + ticker_index),
                "form_type": "10-K",
                "filing_date": "2024-01-15",
                "period_end": "2023-12-31",
                "accession_number": f"{ticker}-2024-10K",
                "filing_url": f"https://example.test/{ticker}/10k",
            }
        )

        data["price_evidence"].append(
            {
                "evidence_id": f"{ticker}_PRICE_VOLUME_JAN2024",
                "ticker": ticker,
                "source_type": "price_volume",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "numeric_facts": feature,
                "text": f"{ticker} January price-volume evidence.",
            }
        )

        data["filings_chunks"].append(
            {
                "chunk_id": f"{ticker}_SEC_XBRL_SUMMARY_ASOF_2024-01-31",
                "ticker": ticker,
                "company_name": COMPANY[ticker],
                "source_type": "SEC_XBRL_COMPANYFACTS",
                "as_of_date": "2024-01-31",
                "text": f"{ticker} filing-derived summary.",
            }
        )

        data["news_articles"].append(
            {
                "news_id": f"{ticker}_NEWS_202401",
                "ticker": ticker,
                "company_name": COMPANY[ticker],
                "date": "2024-01-20",
                "source": "GDELT",
                "domain": "example.test",
                "language": "English",
                "source_country": "United States",
                "title": f"{ticker} business outlook update",
                "body": f"News body for {ticker} covering business momentum.",
                "url": f"https://example.test/news/{ticker}",
            }
        )

    for index, day in enumerate(portfolio_dates):
        data["risk_free"].append(
            {
                "date": day,
                "DGS1MO": f"{5.00 + 0.01 * (index % 5):.6f}",
            }
        )

    for ticker in TICKERS:
        for risk_profile in RISK_PROFILES:
            for agent in AGENTS:
                decision = _decision_for(agent, risk_profile, ticker)

                if agent == "valuation":
                    evidence_id = f"{ticker}_PRICE_VOLUME_JAN2024"
                    source = "price_volume"
                elif agent == "fundamental":
                    evidence_id = f"{ticker}_SEC_XBRL_SUMMARY_ASOF_2024-01-31"
                    source = "SEC_XBRL_COMPANYFACTS"
                elif agent == "sentiment":
                    evidence_id = f"{ticker}_NEWS_202401"
                    source = "GDELT_NEWS"
                else:
                    evidence_id = f"{ticker}_{risk_profile}_DEBATE"
                    source = "debate_log"

                confidence = (
                    0.60
                    + 0.03 * AGENTS.index(agent)
                    + 0.02 * RISK_PROFILES.index(risk_profile)
                )

                report_id = f"{ticker}_{risk_profile}_{agent}_report"
                claim_id = f"{ticker}_{risk_profile}_{agent}_claim_001"

                data["agent_decisions"].append(
                    {
                        "date": "2024-02-01",
                        "ticker": ticker,
                        "company_name": COMPANY[ticker],
                        "agent_type": agent,
                        "risk_profile": risk_profile,
                        "decision": decision,
                        "confidence": f"{confidence:.6f}",
                        "evidence_ids": evidence_id,
                    }
                )

                data["agent_reports"].append(
                    {
                        "report_id": report_id,
                        "ticker": ticker,
                        "company_name": COMPANY[ticker],
                        "agent_type": agent,
                        "risk_profile": risk_profile,
                        "report_text": (
                            f"{agent} recommends {decision} for {ticker} "
                            f"under {risk_profile}."
                        ),
                        "claim_ids": [claim_id],
                        "final_recommendation": decision,
                        "confidence": confidence,
                    }
                )

                data["claim_evidence_links"].append(
                    {
                        "claim_id": claim_id,
                        "report_id": report_id,
                        "ticker": ticker,
                        "agent_type": agent,
                        "claim_type": "recommendation_support",
                        "claim_text": (
                            f"{agent} recommends {decision} for {ticker} "
                            f"under {risk_profile}."
                        ),
                        "evidence_id": evidence_id,
                        "evidence_source": source,
                        "expected_status": "supported",
                    }
                )

    for ticker in TICKERS:
        for risk_profile in RISK_PROFILES:
            debate_id = f"{ticker}_{risk_profile}_DEBATE"
            second_round_messages = []

            for round_number in [1, 2]:
                for agent in SPECIALIST_AGENTS:
                    speaker = f"{agent}_agent"
                    message_id = f"{debate_id}_r{round_number}_{agent}"

                    if agent == "valuation":
                        evidence_ids = [f"{ticker}_PRICE_VOLUME_JAN2024"]
                    elif agent == "fundamental":
                        evidence_ids = [f"{ticker}_SEC_XBRL_SUMMARY_ASOF_2024-01-31"]
                    else:
                        evidence_ids = [f"{ticker}_NEWS_202401"]

                    data["debate_logs"].append(
                        {
                            "debate_id": debate_id,
                            "message_id": message_id,
                            "ticker": ticker,
                            "risk_profile": risk_profile,
                            "round": round_number,
                            "speaker": speaker,
                            "message": (
                                f"{speaker} says "
                                f"{_decision_for(agent, risk_profile, ticker)} "
                                f"for {ticker}."
                            ),
                            "explicit_decision": _decision_for(
                                agent, risk_profile, ticker
                            ),
                            "confidence": 0.70 + 0.01 * round_number,
                            "evidence_ids": evidence_ids,
                            "reply_to": (
                                []
                                if round_number == 1
                                else [f"{debate_id}_r1_{agent}"]
                            ),
                        }
                    )

                    if round_number == 2:
                        second_round_messages.append(message_id)

            terminal_decision = _decision_for("multi_agent", risk_profile, ticker)

            data["debate_logs"].append(
                {
                    "debate_id": debate_id,
                    "message_id": f"{debate_id}_r3_manager",
                    "ticker": ticker,
                    "risk_profile": risk_profile,
                    "round": 3,
                    "speaker": "group_chat_manager",
                    "message": f"Consensus decision is {terminal_decision}. TERMINATE",
                    "explicit_decision": terminal_decision,
                    "confidence": 0.91,
                    "evidence_ids": [
                        f"{ticker}_{risk_profile}_valuation_report",
                        f"{ticker}_{risk_profile}_fundamental_report",
                        f"{ticker}_{risk_profile}_sentiment_report",
                    ],
                    "reply_to": second_round_messages,
                }
            )

    return data


def _write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_dataset(root, data, output_name="output.json"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    (root / "config.json").write_text(
        json.dumps(data["config"], indent=2),
        encoding="utf-8",
    )

    (root / "feature_schema.json").write_text(
        json.dumps(data["feature_schema"], indent=2),
        encoding="utf-8",
    )

    _write_csv(
        root / "universe.csv",
        data["universe"],
        ["ticker", "company_name", "sector", "industry", "benchmark_weight"],
    )

    _write_csv(
        root / "prices.csv",
        data["prices"],
        ["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"],
    )

    _write_csv(
        root / "risk_free.csv",
        data["risk_free"],
        ["date", "DGS1MO"],
    )

    _write_csv(
        root / "stock_features.csv",
        data["stock_features"],
        [
            "ticker",
            "january_return",
            "january_annualized_volatility",
            "january_max_drawdown",
            "january_beta_to_equal_weight_universe",
            "january_average_volume",
        ],
    )

    _write_csv(
        root / "filings_metadata.csv",
        data["filings_metadata"],
        [
            "ticker",
            "company_name",
            "cik",
            "form_type",
            "filing_date",
            "period_end",
            "accession_number",
            "filing_url",
        ],
    )

    _write_csv(
        root / "agent_decisions.csv",
        data["agent_decisions"],
        [
            "date",
            "ticker",
            "company_name",
            "agent_type",
            "risk_profile",
            "decision",
            "confidence",
            "evidence_ids",
        ],
    )

    _write_csv(
        root / "claim_evidence_links.csv",
        data["claim_evidence_links"],
        [
            "claim_id",
            "report_id",
            "ticker",
            "agent_type",
            "claim_type",
            "claim_text",
            "evidence_id",
            "evidence_source",
            "expected_status",
        ],
    )

    _write_jsonl(root / "price_evidence.jsonl", data["price_evidence"])
    _write_jsonl(root / "filings_chunks.jsonl", data["filings_chunks"])
    _write_jsonl(root / "news_articles.jsonl", data["news_articles"])
    _write_jsonl(root / "agent_reports.jsonl", data["agent_reports"])
    _write_jsonl(root / "debate_logs.jsonl", data["debate_logs"])

    return {
        "universe_path": str(root / "universe.csv"),
        "config_path": str(root / "config.json"),
        "feature_schema_path": str(root / "feature_schema.json"),
        "prices_path": str(root / "prices.csv"),
        "risk_free_path": str(root / "risk_free.csv"),
        "stock_features_path": str(root / "stock_features.csv"),
        "price_evidence_path": str(root / "price_evidence.jsonl"),
        "filings_metadata_path": str(root / "filings_metadata.csv"),
        "filings_chunks_path": str(root / "filings_chunks.jsonl"),
        "news_articles_path": str(root / "news_articles.jsonl"),
        "agent_reports_path": str(root / "agent_reports.jsonl"),
        "claim_evidence_links_path": str(root / "claim_evidence_links.csv"),
        "debate_logs_path": str(root / "debate_logs.jsonl"),
        "agent_decisions_path": str(root / "agent_decisions.csv"),
        "output_path": str(root / output_name),
    }


def _run_valid(fn, data=None, output_name="output.json", tie_tol=1e-12):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        payload = _make_valid_dataset() if data is None else data

        kwargs = _write_dataset(root, payload, output_name=output_name)
        kwargs["tie_tol"] = tie_tol

        result = _call_valid(fn, **kwargs)
        saved = json.loads(Path(kwargs["output_path"]).read_text(encoding="utf-8"))

        assert saved == result
        return result, payload, kwargs


def _mutate_and_expect_value_error(fn, mutate, tie_tol=1e-12):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = _make_valid_dataset()

        mutate(data)

        kwargs = _write_dataset(root, data)
        kwargs["tie_tol"] = tie_tol

        _assert_value_error_nonempty(fn, **kwargs)


def _selected(agent, risk_profile):
    return [
        ticker
        for ticker in TICKERS
        if _decision_for(agent, risk_profile, ticker) == "BUY"
    ]


def _feature_means(agent, risk_profile):
    selected = _selected(agent, risk_profile)

    return {
        key: _round12(sum(FEATURES[ticker][key] for ticker in selected) / len(selected))
        for key in [
            "january_return",
            "january_annualized_volatility",
            "january_max_drawdown",
            "january_beta_to_equal_weight_universe",
            "january_average_volume",
        ]
    }


def _jaccard(left, right):
    left = set(left)
    right = set(right)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _price_frame(data):
    frame = pd.DataFrame(data["prices"])
    frame["date"] = pd.to_datetime(frame["date"])

    for column in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        frame[column] = frame[column].astype(float)

    return frame


def _risk_free(data, dates):
    frame = pd.DataFrame(data["risk_free"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["daily"] = frame["DGS1MO"].astype(float) / (
        100.0 * TRADING_DAYS_PER_YEAR
    )

    return (
        frame.set_index("date")["daily"]
        .reindex(dates)
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )


def _sample_std(values):
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _portfolio_expected(data, agent, risk_profile):
    prices = _price_frame(data)
    start = pd.Timestamp(data["config"]["portfolio_start"])
    end = pd.Timestamp(data["config"]["portfolio_end"])

    dates = sorted(
        prices.loc[(prices["date"] >= start) & (prices["date"] <= end), "date"].unique()
    )

    returns_by_ticker = {}

    for ticker in TICKERS:
        series = (
            prices[prices["ticker"] == ticker]
            .set_index("date")
            .loc[dates, "adjusted_close"]
        )
        returns_by_ticker[ticker] = series.pct_change().dropna().to_numpy(dtype=float)

    selected = [
        row["ticker"]
        for row in data["agent_decisions"]
        if row["agent_type"] == agent
        and row["risk_profile"] == risk_profile
        and row["decision"] == "BUY"
    ]

    portfolio_returns = np.mean(
        [returns_by_ticker[ticker] for ticker in selected],
        axis=0,
    )

    benchmark_returns = np.mean(
        [returns_by_ticker[ticker] for ticker in TICKERS],
        axis=0,
    )

    risk_free = _risk_free(data, dates)[1:]
    excess = portfolio_returns - risk_free
    wealth = np.cumprod(1.0 + portfolio_returns)

    rolling = []

    for end_index in range(ROLLING_SHARPE_WINDOW, len(portfolio_returns) + 1):
        window = excess[end_index - ROLLING_SHARPE_WINDOW:end_index]
        standard_deviation = _sample_std(window)

        rolling.append(
            None
            if standard_deviation <= 1e-12
            else float(
                np.mean(window)
                / standard_deviation
                * math.sqrt(TRADING_DAYS_PER_YEAR)
            )
        )

    finite_rolling = [value for value in rolling if value is not None]

    design = sm.add_constant(benchmark_returns - risk_free, has_constant="add")
    fit = sm.OLS(excess, design).fit()

    alpha = float(fit.params[0])
    residual_std = math.sqrt(
        float(np.sum(np.asarray(fit.resid) ** 2) / (len(fit.resid) - 2))
    )

    benchmark_cumulative = float(np.prod(1.0 + benchmark_returns) - 1.0)
    cumulative = float(np.prod(1.0 + portfolio_returns) - 1.0)
    sharpe_std = _sample_std(excess)

    return {
        "selected_tickers": selected,
        "selected_count": len(selected),
        "cumulative_return": _round12(cumulative),
        "annualized_volatility": _round12(
            math.sqrt(TRADING_DAYS_PER_YEAR) * _sample_std(portfolio_returns)
        ),
        "max_drawdown": _round12(
            float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))
        ),
        "sharpe_ratio": (
            None
            if sharpe_std <= 1e-12
            else _round12(
                float(
                    np.mean(excess)
                    / sharpe_std
                    * math.sqrt(TRADING_DAYS_PER_YEAR)
                )
            )
        ),
        "benchmark_excess_return": _round12(cumulative - benchmark_cumulative),
        "rolling_sharpe_min": (
            _round12(min(finite_rolling)) if finite_rolling else None
        ),
        "rolling_sharpe_max": (
            _round12(max(finite_rolling)) if finite_rolling else None
        ),
        "rolling_sharpe_last": _round12(rolling[-1]) if rolling else None,
        "ols_excess_return_model": {
            "alpha": _round12(alpha),
            "beta": _round12(float(fit.params[1])),
            "r_squared": (
                _round12(float(fit.rsquared))
                if math.isfinite(float(fit.rsquared))
                else 0.0
            ),
            "residual_std": _round12(residual_std),
            "trend_label": (
                "positive_alpha"
                if alpha > 1e-12
                else ("negative_alpha" if alpha < -1e-12 else "flat_alpha")
            ),
        },
    }


def _benchmark_expected(data):
    prices = _price_frame(data)
    start = pd.Timestamp(data["config"]["portfolio_start"])
    end = pd.Timestamp(data["config"]["portfolio_end"])

    dates = sorted(
        prices.loc[(prices["date"] >= start) & (prices["date"] <= end), "date"].unique()
    )

    returns = []

    for ticker in TICKERS:
        series = (
            prices[prices["ticker"] == ticker]
            .set_index("date")
            .loc[dates, "adjusted_close"]
        )
        returns.append(series.pct_change().dropna().to_numpy(dtype=float))

    benchmark_returns = np.mean(returns, axis=0)
    risk_free = _risk_free(data, dates)[1:]
    excess = benchmark_returns - risk_free
    wealth = np.cumprod(1.0 + benchmark_returns)
    excess_std = _sample_std(excess)

    return {
        "cumulative_return": _round12(np.prod(1.0 + benchmark_returns) - 1.0),
        "annualized_volatility": _round12(
            math.sqrt(TRADING_DAYS_PER_YEAR) * _sample_std(benchmark_returns)
        ),
        "max_drawdown": _round12(
            float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))
        ),
        "sharpe_ratio": (
            None
            if excess_std <= 1e-12
            else _round12(
                float(
                    np.mean(excess)
                    / excess_std
                    * math.sqrt(TRADING_DAYS_PER_YEAR)
                )
            )
        ),
    }


def _rank_order(values, higher=True):
    def canonical_index(label):
        agent, risk_profile = label.split("|")
        return AGENTS.index(agent), RISK_PROFILES.index(risk_profile)

    ranks = {}

    for label, value in values.items():
        rank = 1

        for other_label, other_value in values.items():
            if other_label == label:
                continue

            if higher and other_value > value + 1e-12:
                rank += 1

            if not higher and other_value < value - 1e-12:
                rank += 1

        ranks[label] = rank

    return sorted(
        values,
        key=lambda label: (ranks[label], *canonical_index(label)),
    )


def _set_decision_and_report(data, agent, risk_profile, ticker, decision):
    for row in data["agent_decisions"]:
        if (
            row["agent_type"] == agent
            and row["risk_profile"] == risk_profile
            and row["ticker"] == ticker
        ):
            row["decision"] = decision

    for row in data["agent_reports"]:
        if (
            row["agent_type"] == agent
            and row["risk_profile"] == risk_profile
            and row["ticker"] == ticker
        ):
            row["final_recommendation"] = decision


def test_case_1_valid_dataset_profile_output_and_json_types():
    fn = multi_agent_equity_consistency_audit

    result, data, _ = _run_valid(fn)

    assert list(result.keys()) == EXPECTED_TOP_LEVEL_KEYS

    assert result["dataset_profile"] == {
        "ticker_count": 5,
        "agent_count": 4,
        "risk_profile_count": 3,
        "decision_count": 60,
        "report_count": 60,
        "claim_count": 60,
        "debate_count": 15,
        "price_observation_count": len(data["prices"]),
        "portfolio_trading_day_count": (
            len(_business_dates("2024-02-01", "2024-03-15")) - 1
        ),
    }

    _assert_json_native(result)


def test_case_2_evidence_claim_counts_agent_scores_and_claim_level_mapping():
    fn = multi_agent_equity_consistency_audit

    result, _, _ = _run_valid(fn)

    assert result["evidence_audit"]["claim_status_counts"] == {
        "supported": 60,
        "missing_evidence": 0,
        "role_access_violation": 0,
        "decision_mismatch": 0,
        "unsupported": 0,
    }

    for agent in AGENTS:
        assert result["evidence_audit"]["agent_claim_scores"][agent] == {
            "claim_count": 15,
            "supported_count": 15,
            "support_rate": 1.0,
            "violation_count": 0,
        }

    first = result["evidence_audit"]["claim_level_results"][
        "AAA_risk_neutral_valuation_claim_001"
    ]

    assert first == {
        "ticker": "AAA",
        "agent": "valuation",
        "risk_profile": "risk_neutral",
        "status": "supported",
        "evidence_id": "AAA_PRICE_VOLUME_JAN2024",
        "evidence_source": "price_volume",
    }


def test_case_3_debate_graph_page_rank_and_global_debate_score():
    fn = multi_agent_equity_consistency_audit

    result, data, _ = _run_valid(fn)

    debate_id = "AAA_risk_neutral_DEBATE"
    debate = result["debate_audit"]["debate_level_results"][debate_id]

    rows = [row for row in data["debate_logs"] if row["debate_id"] == debate_id]

    graph = nx.DiGraph()

    for row in rows:
        graph.add_node(row["message_id"], speaker=row["speaker"])

    for row in rows:
        for parent in row["reply_to"]:
            graph.add_edge(parent, row["message_id"])

    pagerank = nx.pagerank(graph)
    speaker_scores = {}

    for node, score in pagerank.items():
        speaker = graph.nodes[node]["speaker"]
        speaker_scores[speaker] = speaker_scores.get(speaker, 0.0) + score

    assert debate["node_count"] == 7
    assert debate["edge_count"] == 6
    assert debate["weak_component_count"] == 1
    assert debate["turn_count_by_specialist"] == {
        "valuation": 2,
        "fundamental": 2,
        "sentiment": 2,
    }
    assert debate["terminal_decision"] == _decision_for(
        "multi_agent",
        "risk_neutral",
        "AAA",
    )
    assert debate["termination_valid"] is True
    assert debate["consensus_matches_decision_table"] is True

    assert result["debate_audit"]["global_debate_score"] == 1.0
    assert result["debate_audit"]["invalid_debate_count"] == 0

    for speaker, score in speaker_scores.items():
        assert debate["speaker_pagerank"][speaker] == _round12(score)


def test_case_4_portfolio_metrics_benchmark_ols_and_rolling_sharpe():
    fn = multi_agent_equity_consistency_audit

    result, data, _ = _run_valid(fn)

    keys = [
        "valuation|risk_neutral",
        "fundamental|risk_averse",
        "sentiment|risk_seeking",
        "multi_agent|risk_neutral",
    ]

    for key in keys:
        agent, risk_profile = key.split("|")
        expected = _portfolio_expected(data, agent, risk_profile)
        observed = result["portfolio_audit"]["portfolio_metrics"][key]

        _assert_nested_close(observed, expected)

    _assert_nested_close(
        result["portfolio_audit"]["benchmark_metrics"],
        _benchmark_expected(data),
    )


def test_case_5_portfolio_rankings_follow_metric_direction_and_canonical_ties():
    fn = multi_agent_equity_consistency_audit

    result, _, _ = _run_valid(fn)
    metrics = result["portfolio_audit"]["portfolio_metrics"]

    assert result["portfolio_audit"]["rankings"]["by_cumulative_return_desc"] == (
        _rank_order({key: value["cumulative_return"] for key, value in metrics.items()}, True)
    )

    assert result["portfolio_audit"]["rankings"]["by_sharpe_desc"] == (
        _rank_order({key: value["sharpe_ratio"] for key, value in metrics.items()}, True)
    )

    assert result["portfolio_audit"]["rankings"]["by_drawdown_asc"] == (
        _rank_order({key: abs(value["max_drawdown"]) for key, value in metrics.items()}, False)
    )

    assert result["portfolio_audit"]["rankings"]["by_volatility_asc"] == (
        _rank_order(
            {key: value["annualized_volatility"] for key, value in metrics.items()},
            False,
        )
    )

    tied, _, _ = _run_valid(fn, tie_tol=1.0)
    expected_order = [f"{agent}|{risk}" for agent in AGENTS for risk in RISK_PROFILES]

    assert tied["portfolio_audit"]["rankings"]["by_cumulative_return_desc"] == expected_order
    assert tied["portfolio_audit"]["rankings"]["by_volatility_asc"] == expected_order


def test_case_6_risk_profile_sets_jaccard_feature_means_and_projection():
    fn = multi_agent_equity_consistency_audit

    result, _, _ = _run_valid(fn)

    for agent in AGENTS:
        profile = result["risk_profile_audit"]["agent_risk_profiles"][agent]

        selected_sets = {
            risk_profile: _selected(agent, risk_profile)
            for risk_profile in RISK_PROFILES
        }

        assert profile["selected_sets"] == selected_sets

        assert profile["jaccard"] == {
            "risk_neutral|risk_averse": _round12(
                _jaccard(
                    selected_sets["risk_neutral"],
                    selected_sets["risk_averse"],
                )
            ),
            "risk_neutral|risk_seeking": _round12(
                _jaccard(
                    selected_sets["risk_neutral"],
                    selected_sets["risk_seeking"],
                )
            ),
            "risk_averse|risk_seeking": _round12(
                _jaccard(
                    selected_sets["risk_averse"],
                    selected_sets["risk_seeking"],
                )
            ),
        }

        for risk_profile in RISK_PROFILES:
            assert profile["risk_feature_means"][risk_profile] == (
                _feature_means(agent, risk_profile)
            )

        assert profile["monotonicity_violations"] == []

        projection = profile["cvxpy_projection"]

        assert projection["solver"] in {"OSQP", "SCS"}
        assert projection["status"] in {"optimal", "optimal_inaccurate"}
        assert projection["projected_average_volatility"] <= (
            _feature_means(agent, "risk_neutral")["january_annualized_volatility"]
            + 1e-8
        )


def test_case_7_summary_clean_signature_and_scores():
    fn = multi_agent_equity_consistency_audit

    result, _, _ = _run_valid(fn)

    assert result["summary"]["all_claims_supported_or_missing_by_design"] is True
    assert result["summary"]["all_debates_valid"] is True
    assert result["summary"]["all_consensus_decisions_reconciled"] is True
    assert result["summary"]["risk_averse_has_lower_average_volatility_than_risk_neutral"] is True
    assert result["summary"]["cvxpy_projection_all_feasible"] is True
    assert result["summary"]["global_consistency_score"] == 1.0
    assert result["summary"]["audit_signature"] == "fully_consistent"
    assert result["summary"]["audit_signature_code"] == 3


def test_case_8_determinism_whitespace_numeric_strings_and_saved_output():
    fn = multi_agent_equity_consistency_audit

    base, _, _ = _run_valid(fn, output_name="base.json")

    data = _make_valid_dataset()

    for section in [
        "universe",
        "prices",
        "risk_free",
        "stock_features",
        "filings_metadata",
        "agent_decisions",
        "claim_evidence_links",
    ]:
        for row in data[section]:
            for key, value in list(row.items()):
                if isinstance(value, str):
                    row[key] = f"  {value}  "

    for section in [
        "price_evidence",
        "filings_chunks",
        "news_articles",
        "agent_reports",
        "debate_logs",
    ]:
        for row in data[section]:
            for key, value in list(row.items()):
                if isinstance(value, str):
                    row[key] = f"  {value}  "

    assert _run_valid(fn, data=data, output_name="strings.json")[0] == base
    assert _run_valid(fn, data=_make_valid_dataset(), output_name="repeat.json")[0] == base

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        kwargs = _write_dataset(root / "input", _make_valid_dataset())
        kwargs["output_path"] = str(root / "nested" / "custom_output.json")
        kwargs["tie_tol"] = 1e-12

        result = _call_valid(fn, **kwargs)

        assert json.loads(Path(kwargs["output_path"]).read_text(encoding="utf-8")) == result


def test_case_9_risk_free_forward_backward_fill_is_used():
    fn = multi_agent_equity_consistency_audit

    constant_data = _make_valid_dataset()

    for row in constant_data["risk_free"]:
        row["DGS1MO"] = "5.000000"

    expected = _run_valid(fn, data=constant_data, output_name="constant.json")[0]

    sparse_data = copy.deepcopy(constant_data)
    sparse_data["risk_free"] = [
        row
        for index, row in enumerate(sparse_data["risk_free"])
        if index in {3, 10, 20}
    ]

    observed = _run_valid(fn, data=sparse_data, output_name="filled.json")[0]

    assert observed["portfolio_audit"] == expected["portfolio_audit"]


def test_case_10_invalid_tie_tol_malformed_files_and_top_levels_raise_value_error():
    fn = multi_agent_equity_consistency_audit

    bad_tolerances = [
        True,
        False,
        0,
        -1e-12,
        float("nan"),
        float("inf"),
        "",
        None,
    ]

    for bad_tol in bad_tolerances:
        with tempfile.TemporaryDirectory() as td:
            kwargs = _write_dataset(Path(td), _make_valid_dataset())
            kwargs["tie_tol"] = bad_tol
            _assert_value_error_nonempty(fn, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        kwargs = _write_dataset(Path(td), _make_valid_dataset())
        Path(kwargs["config_path"]).write_text("{bad json", encoding="utf-8")
        _assert_value_error_nonempty(fn, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        kwargs = _write_dataset(Path(td), _make_valid_dataset())
        Path(kwargs["price_evidence_path"]).write_text(
            '{"evidence_id": "X"}\n{bad json}\n',
            encoding="utf-8",
        )
        _assert_value_error_nonempty(fn, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        kwargs = _write_dataset(Path(td), _make_valid_dataset())
        Path(kwargs["config_path"]).write_text("[]", encoding="utf-8")
        _assert_value_error_nonempty(fn, **kwargs)


def test_case_11_config_and_feature_schema_contract_validation():
    fn = multi_agent_equity_consistency_audit

    mutations = [
        lambda data: data["config"].pop("analysis_start"),
        lambda data: data["config"].update({"portfolio_start": "2024-01-15"}),
        lambda data: data["config"].update({"trading_days_per_year": 365}),
        lambda data: data["config"].update({"rolling_sharpe_window": 10}),
        lambda data: data["config"].update({"portfolio_weighting": "cap_weight"}),
        lambda data: data["config"].update({"hold_policy": "include"}),
        lambda data: data["config"].update(
            {"allowed_agents": ["valuation", "sentiment", "fundamental", "multi_agent"]}
        ),
        lambda data: data["feature_schema"].pop("valuation"),
        lambda data: data["feature_schema"].update({"macro": ["ticker"]}),
        lambda data: data["feature_schema"].update({"valuation": "ticker"}),
    ]

    for mutate in mutations:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_12_duplicate_ids_ticker_coverage_and_required_columns():
    fn = multi_agent_equity_consistency_audit

    mutations = [
        lambda data: data["universe"].append(copy.deepcopy(data["universe"][0])),
        lambda data: data["stock_features"].pop(),
        lambda data: data["prices"].pop(),
        lambda data: data["filings_metadata"][0].update({"ticker": "ZZZ"}),
        lambda data: data["agent_decisions"].append(copy.deepcopy(data["agent_decisions"][0])),
        lambda data: data["agent_reports"][0].update(
            {"report_id": data["agent_reports"][1]["report_id"]}
        ),
        lambda data: data["claim_evidence_links"].append(
            copy.deepcopy(data["claim_evidence_links"][0])
        ),
    ]

    for mutate in mutations:
        _mutate_and_expect_value_error(fn, mutate)

    with tempfile.TemporaryDirectory() as td:
        kwargs = _write_dataset(Path(td), _make_valid_dataset())
        frame = pd.read_csv(kwargs["prices_path"], dtype=str).drop(
            columns=["adjusted_close"]
        )
        frame.to_csv(kwargs["prices_path"], index=False)
        _assert_value_error_nonempty(fn, **kwargs)


def test_case_13_numeric_and_date_validation():
    fn = multi_agent_equity_consistency_audit

    mutations = [
        lambda data: data["prices"][0].update({"adjusted_close": "0"}),
        lambda data: data["prices"][0].update({"open": "-1"}),
        lambda data: data["prices"][0].update({"volume": "-100"}),
        lambda data: data["prices"][0].update({"close": True}),
        lambda data: data["stock_features"][0].update(
            {"january_annualized_volatility": "nan"}
        ),
        lambda data: data["agent_decisions"][0].update({"confidence": "1.5"}),
        lambda data: data["agent_reports"][0].update({"confidence": "nan"}),
        lambda data: data["risk_free"][0].update({"DGS1MO": "inf"}),
        lambda data: data["prices"][0].update({"date": "2024-99-99"}),
        lambda data: data["price_evidence"][0].update({"end_date": "2024-02-15"}),
        lambda data: data["filings_chunks"][0].update({"as_of_date": "2024-02-15"}),
        lambda data: data["news_articles"][0].update({"date": "2024-04-01"}),
    ]

    for mutate in mutations:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_14_decision_table_labels_coverage_and_evidence_ids():
    fn = multi_agent_equity_consistency_audit

    mutations = [
        lambda data: data["agent_decisions"][0].update({"decision": "STRONG_BUY"}),
        lambda data: data["agent_decisions"][0].update({"agent_type": "macro"}),
        lambda data: data["agent_decisions"][0].update({"risk_profile": "risk_low"}),
        lambda data: data["agent_decisions"][0].update({"ticker": "ZZZ"}),
        lambda data: data["agent_decisions"].pop(),
        lambda data: data["agent_decisions"][0].update(
            {"evidence_ids": "MISSING_EVIDENCE_ID"}
        ),
    ]

    for mutate in mutations:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_15_evidence_identifier_and_required_key_validation():
    fn = multi_agent_equity_consistency_audit

    mutations = [
        lambda data: data["price_evidence"].append(copy.deepcopy(data["price_evidence"][0])),
        lambda data: data["filings_chunks"][0].update(
            {"chunk_id": data["price_evidence"][0]["evidence_id"]}
        ),
        lambda data: data["news_articles"][0].update(
            {"news_id": data["price_evidence"][0]["evidence_id"]}
        ),
        lambda data: data["price_evidence"][0].update({"ticker": "ZZZ"}),
        lambda data: data["filings_chunks"][0].pop("chunk_id"),
        lambda data: data["news_articles"][0].pop("body"),
    ]

    for mutate in mutations:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_16_claim_status_classes_missing_role_violation_mismatch_and_unsupported():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()
    data["claim_evidence_links"][0]["evidence_id"] = "EXPLICITLY_MISSING_EVIDENCE"
    data["claim_evidence_links"][0]["expected_status"] = "missing_evidence"

    result = _run_valid(fn, data=data)[0]

    assert result["evidence_audit"]["claim_status_counts"]["missing_evidence"] == 1

    data = _make_valid_dataset()
    claim = next(
        row for row in data["claim_evidence_links"] if row["agent_type"] == "valuation"
    )
    claim["evidence_id"] = f"{claim['ticker']}_SEC_XBRL_SUMMARY_ASOF_2024-01-31"
    claim["evidence_source"] = "SEC_XBRL_COMPANYFACTS"

    result = _run_valid(fn, data=data)[0]

    assert result["evidence_audit"]["claim_level_results"][claim["claim_id"]]["status"] == (
        "role_access_violation"
    )

    data = _make_valid_dataset()
    data["agent_reports"][0]["final_recommendation"] = (
        "SELL" if data["agent_reports"][0]["final_recommendation"] != "SELL" else "BUY"
    )

    result = _run_valid(fn, data=data)[0]

    assert result["evidence_audit"]["claim_status_counts"]["decision_mismatch"] == 1

    data = _make_valid_dataset()
    data["claim_evidence_links"][0]["evidence_source"] = "wrong_source"

    result = _run_valid(fn, data=data)[0]

    assert result["evidence_audit"]["claim_status_counts"]["unsupported"] == 1


def test_case_17_missing_unallowed_claim_evidence_and_report_claim_mismatch_raise_value_error():
    fn = multi_agent_equity_consistency_audit

    def missing_unallowed(data):
        data["claim_evidence_links"][0]["evidence_id"] = "MISSING_EVIDENCE"
        data["claim_evidence_links"][0]["expected_status"] = "supported"

    def report_claim_mismatch(data):
        data["agent_reports"][0]["claim_ids"] = ["undeclared_claim_id"]

    for mutate in [missing_unallowed, report_claim_mismatch]:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_18_debate_graph_and_message_reference_validation():
    fn = multi_agent_equity_consistency_audit

    mutations = [
        lambda data: data["debate_logs"][1].update({"reply_to": ["NO_SUCH_MESSAGE"]}),
        lambda data: data["debate_logs"][3].update(
            {
                "reply_to": [data["debate_logs"][0]["message_id"]],
                "round": 1,
            }
        ),
        lambda data: data["debate_logs"][3].update({"round": 0}),
        lambda data: data["debate_logs"][0].update(
            {"message_id": data["debate_logs"][1]["message_id"]}
        ),
        lambda data: data["debate_logs"][0].update({"explicit_decision": "MAYBE"}),
        lambda data: data["debate_logs"][0].update({"confidence": "-0.1"}),
    ]

    for mutate in mutations:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_19_debate_evidence_references_and_turn_coverage_are_enforced():
    fn = multi_agent_equity_consistency_audit

    def bad_debate_evidence(data):
        data["debate_logs"][0]["evidence_ids"] = ["NO_SUCH_EVIDENCE"]

    def remove_specialist_second_turn(data):
        data["debate_logs"] = [
            row
            for row in data["debate_logs"]
            if row["message_id"] != "AAA_risk_neutral_DEBATE_r2_valuation"
        ]

    def remove_entire_debate(data):
        data["debate_logs"] = [
            row
            for row in data["debate_logs"]
            if row["debate_id"] != "AAA_risk_neutral_DEBATE"
        ]

    for mutate in [
        bad_debate_evidence,
        remove_specialist_second_turn,
        remove_entire_debate,
    ]:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_20_invalid_terminal_consensus_and_terminal_token_are_scored():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()
    manager = next(
        row
        for row in data["debate_logs"]
        if row["message_id"] == "AAA_risk_neutral_DEBATE_r3_manager"
    )

    manager["explicit_decision"] = (
        "SELL" if manager["explicit_decision"] != "SELL" else "BUY"
    )

    result = _run_valid(fn, data=data)[0]
    debate = result["debate_audit"]["debate_level_results"]["AAA_risk_neutral_DEBATE"]

    assert debate["termination_valid"] is True
    assert debate["consensus_matches_decision_table"] is False
    assert result["debate_audit"]["invalid_debate_count"] == 1
    assert result["summary"]["all_consensus_decisions_reconciled"] is False

    data = _make_valid_dataset()
    manager = next(
        row
        for row in data["debate_logs"]
        if row["message_id"] == "AAA_risk_neutral_DEBATE_r3_manager"
    )

    manager["message"] = "Consensus without configured token."

    result = _run_valid(fn, data=data)[0]

    assert result["debate_audit"]["debate_level_results"]["AAA_risk_neutral_DEBATE"][
        "termination_valid"
    ] is False


def test_case_21_unresolved_dissent_flag_changes_with_specialist_final_decisions():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    terminal = _decision_for("multi_agent", "risk_neutral", "AAA")
    dissent = "SELL" if terminal != "SELL" else "BUY"

    for agent in ["valuation", "fundamental"]:
        row = next(
            item
            for item in data["debate_logs"]
            if item["message_id"] == f"AAA_risk_neutral_DEBATE_r2_{agent}"
        )
        row["explicit_decision"] = dissent

    result = _run_valid(fn, data=data)[0]

    assert result["debate_audit"]["debate_level_results"]["AAA_risk_neutral_DEBATE"][
        "unresolved_dissent"
    ] is True


def test_case_22_empty_portfolio_missing_prices_and_short_window_raise_value_error():
    fn = multi_agent_equity_consistency_audit

    def empty_buy(data):
        for row in data["agent_decisions"]:
            if row["agent_type"] == "valuation" and row["risk_profile"] == "risk_neutral":
                row["decision"] = "SELL"

        for row in data["agent_reports"]:
            if row["agent_type"] == "valuation" and row["risk_profile"] == "risk_neutral":
                row["final_recommendation"] = "SELL"

    def missing_price(data):
        data["prices"] = [
            row
            for row in data["prices"]
            if not (row["ticker"] == "AAA" and row["date"] == "2024-02-05")
        ]

    def short_window(data):
        data["config"].update({"portfolio_end": "2024-02-01"})

    for mutate in [empty_buy, missing_price, short_window]:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_23_zero_excess_return_sets_sharpe_to_none_and_lowers_score():
    import warnings

    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    for row in data["prices"]:
        row["open"] = "100.0"
        row["high"] = "100.0"
        row["low"] = "100.0"
        row["close"] = "100.0"
        row["adjusted_close"] = "100.0"

    for row in data["risk_free"]:
        row["DGS1MO"] = "0.0"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
            module="statsmodels.regression.linear_model",
        )
        result = _run_valid(fn, data=data)[0]

    assert all(
        metrics["sharpe_ratio"] is None
        for metrics in result["portfolio_audit"]["portfolio_metrics"].values()
    )

    assert result["portfolio_audit"]["benchmark_metrics"]["sharpe_ratio"] is None
    assert result["summary"]["global_consistency_score"] < 1.0


def test_case_24_risk_profile_monotonicity_violation_changes_score_and_signature():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    for row in data["agent_decisions"]:
        if row["agent_type"] == "valuation" and row["risk_profile"] == "risk_averse":
            if row["ticker"] == "AAA":
                row["decision"] = "SELL"
            if row["ticker"] == "DDD":
                row["decision"] = "BUY"

    for row in data["agent_reports"]:
        if row["agent_type"] == "valuation" and row["risk_profile"] == "risk_averse":
            if row["ticker"] == "AAA":
                row["final_recommendation"] = "SELL"
            if row["ticker"] == "DDD":
                row["final_recommendation"] = "BUY"

    result = _run_valid(fn, data=data)[0]

    violations = result["risk_profile_audit"]["agent_risk_profiles"]["valuation"][
        "monotonicity_violations"
    ]

    assert "higher_average_volatility" in violations
    assert result["risk_profile_audit"]["total_monotonicity_violation_count"] >= 1
    assert result["summary"]["global_consistency_score"] < 1.0
    assert result["summary"]["audit_signature"] in {
        "portfolio_consistent_with_evidence_warnings",
        "risk_profile_inconsistent",
    }


def test_case_25_output_float_rounding_to_12_decimals():
    fn = multi_agent_equity_consistency_audit

    result = _run_valid(fn)[0]

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, float):
            assert math.isfinite(value)
            assert value == round(value, 12)

    visit(result)


def test_case_26_statsmodels_ols_networkx_digraph_and_cvxpy_are_used():
    fn = multi_agent_equity_consistency_audit

    real_ols = sm.OLS
    real_digraph = nx.DiGraph
    real_solve = cp.Problem.solve

    ols_calls = []
    graphs = []
    solve_calls = []

    def spy_ols(endog, exog, *args, **kwargs):
        ols_calls.append((len(endog), np.asarray(exog).shape))
        return real_ols(endog, exog, *args, **kwargs)

    class SpyDiGraph(real_digraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            graphs.append(self)

    def spy_solve(self, *args, **kwargs):
        solver = kwargs.get("solver", args[0] if args else None)
        solve_calls.append(solver)
        return real_solve(self, *args, **kwargs)

    sm.OLS = spy_ols
    nx.DiGraph = SpyDiGraph
    cp.Problem.solve = spy_solve

    try:
        result = _run_valid(fn)[0]
        rows = result["dataset_profile"]["portfolio_trading_day_count"]

        matching_ols_calls = [
            call for call in ols_calls if call == (rows, (rows, 2))
        ]

        assert len(matching_ols_calls) >= len(AGENTS) * len(RISK_PROFILES)

        assert len(graphs) >= result["dataset_profile"]["debate_count"]
        assert any(
            graph.number_of_nodes() == 7 and graph.number_of_edges() == 6
            for graph in graphs
        )

        assert any(
            str(call).upper().endswith("OSQP") or call == cp.OSQP
            for call in solve_calls
        )

    finally:
        sm.OLS = real_ols
        nx.DiGraph = real_digraph
        cp.Problem.solve = real_solve


def test_case_27_portfolio_price_mutation_changes_only_portfolio_layer():
    fn = multi_agent_equity_consistency_audit

    base = _run_valid(fn, output_name="base.json")[0]

    data = _make_valid_dataset()

    for row in data["prices"]:
        if row["ticker"] == "CCC" and row["date"] >= "2024-02-01":
            for column in ["open", "high", "low", "close", "adjusted_close"]:
                row[column] = str(float(row[column]) * 1.02)

    changed = _run_valid(fn, data=data, output_name="changed.json")[0]

    assert changed["portfolio_audit"] != base["portfolio_audit"]
    assert changed["evidence_audit"] == base["evidence_audit"]
    assert changed["debate_audit"] == base["debate_audit"]
    assert changed["risk_profile_audit"] == base["risk_profile_audit"]


def test_case_28_claim_mutation_changes_evidence_layer_without_changing_portfolios():
    fn = multi_agent_equity_consistency_audit

    base = _run_valid(fn, output_name="base.json")[0]

    data = _make_valid_dataset()
    data["claim_evidence_links"][0]["evidence_source"] = "wrong_source"

    changed = _run_valid(fn, data=data, output_name="changed.json")[0]

    assert changed["evidence_audit"] != base["evidence_audit"]
    assert changed["portfolio_audit"] == base["portfolio_audit"]
    assert changed["risk_profile_audit"] == base["risk_profile_audit"]


def test_case_29_filing_metadata_validation():
    fn = multi_agent_equity_consistency_audit

    mutations = [
        lambda data: data["filings_metadata"][0].update({"form_type": "8-K"}),
        lambda data: data["filings_metadata"][0].update({"cik": "not-int"}),
        lambda data: data["filings_metadata"].append(copy.deepcopy(data["filings_metadata"][0])),
    ]

    for mutate in mutations:
        _mutate_and_expect_value_error(fn, mutate)


def test_case_30_non_jsonl_object_records_are_rejected():
    fn = multi_agent_equity_consistency_audit

    with tempfile.TemporaryDirectory() as td:
        kwargs = _write_dataset(Path(td), _make_valid_dataset())

        Path(kwargs["agent_reports_path"]).write_text(
            '[{"report_id": "array-is-not-a-record-object"}]\n',
            encoding="utf-8",
        )

        _assert_value_error_nonempty(fn, **kwargs)


def test_case_31_buy_only_policy_excludes_hold_and_sell_from_portfolio():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    result = _run_valid(fn, data=data)[0]
    observed = result["portfolio_audit"]["portfolio_metrics"]["valuation|risk_neutral"]

    assert observed["selected_tickers"] == ["AAA", "BBB", "CCC"]
    assert observed["selected_count"] == 3
    assert "EEE" not in observed["selected_tickers"]
    assert "DDD" not in observed["selected_tickers"]

    expected = _portfolio_expected(data, "valuation", "risk_neutral")
    _assert_nested_close(observed, expected)


def test_case_32_rolling_sharpe_values_are_computed_not_placeholders():
    fn = multi_agent_equity_consistency_audit

    result, data, _ = _run_valid(fn)

    expected = _portfolio_expected(data, "multi_agent", "risk_neutral")
    observed = result["portfolio_audit"]["portfolio_metrics"]["multi_agent|risk_neutral"]

    assert observed["rolling_sharpe_min"] == expected["rolling_sharpe_min"]
    assert observed["rolling_sharpe_max"] == expected["rolling_sharpe_max"]
    assert observed["rolling_sharpe_last"] == expected["rolling_sharpe_last"]

    assert not (
        observed["rolling_sharpe_min"] == 0.0
        and observed["rolling_sharpe_max"] == 0.0
        and observed["rolling_sharpe_last"] == 0.0
    )


def test_case_33_ols_residual_std_uses_regression_degrees_of_freedom():
    fn = multi_agent_equity_consistency_audit

    result, data, _ = _run_valid(fn)

    expected = _portfolio_expected(data, "valuation", "risk_neutral")
    observed = result["portfolio_audit"]["portfolio_metrics"]["valuation|risk_neutral"]

    assert observed["ols_excess_return_model"]["residual_std"] == (
        expected["ols_excess_return_model"]["residual_std"]
    )

    prices = _price_frame(data)
    start = pd.Timestamp(data["config"]["portfolio_start"])
    end = pd.Timestamp(data["config"]["portfolio_end"])
    dates = sorted(
        prices.loc[(prices["date"] >= start) & (prices["date"] <= end), "date"].unique()
    )

    returns_by_ticker = {}
    for ticker in TICKERS:
        series = (
            prices[prices["ticker"] == ticker]
            .set_index("date")
            .loc[dates, "adjusted_close"]
        )
        returns_by_ticker[ticker] = series.pct_change().dropna().to_numpy(dtype=float)

    portfolio_returns = np.mean(
        [returns_by_ticker[ticker] for ticker in ["AAA", "BBB", "CCC"]],
        axis=0,
    )
    benchmark_returns = np.mean(
        [returns_by_ticker[ticker] for ticker in TICKERS],
        axis=0,
    )
    risk_free = _risk_free(data, dates)[1:]
    fit = sm.OLS(
        portfolio_returns - risk_free,
        sm.add_constant(benchmark_returns - risk_free, has_constant="add"),
    ).fit()

    wrong_sample_resid_std = _round12(float(np.std(fit.resid, ddof=1)))
    correct_df_resid_std = _round12(
        math.sqrt(float(np.sum(np.asarray(fit.resid) ** 2) / (len(fit.resid) - 2)))
    )

    assert wrong_sample_resid_std != correct_df_resid_std
    assert observed["ols_excess_return_model"]["residual_std"] == correct_df_resid_std


def test_case_34_report_count_is_from_agent_reports_not_claim_rows():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    report = next(
        row
        for row in data["agent_reports"]
        if row["report_id"] == "AAA_risk_neutral_valuation_report"
    )
    report["claim_ids"] = [
        "AAA_risk_neutral_valuation_claim_001",
        "AAA_risk_neutral_valuation_claim_002",
    ]

    data["claim_evidence_links"].append(
        {
            "claim_id": "AAA_risk_neutral_valuation_claim_002",
            "report_id": "AAA_risk_neutral_valuation_report",
            "ticker": "AAA",
            "agent_type": "valuation",
            "claim_type": "additional_price_support",
            "claim_text": "Additional valuation evidence supports the recommendation.",
            "evidence_id": "AAA_PRICE_VOLUME_JAN2024",
            "evidence_source": "price_volume",
            "expected_status": "supported",
        }
    )

    result = _run_valid(fn, data=data)[0]

    assert result["dataset_profile"]["report_count"] == 60
    assert result["dataset_profile"]["claim_count"] == 61
    assert result["evidence_audit"]["claim_status_counts"]["supported"] == 61


def test_case_35_monotonicity_violation_count_counts_each_failed_check():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    _set_decision_and_report(data, "valuation", "risk_averse", "AAA", "SELL")
    _set_decision_and_report(data, "valuation", "risk_averse", "BBB", "SELL")
    _set_decision_and_report(data, "valuation", "risk_averse", "DDD", "BUY")

    result = _run_valid(fn, data=data)[0]

    violations = result["risk_profile_audit"]["agent_risk_profiles"]["valuation"][
        "monotonicity_violations"
    ]

    assert set(violations) == {
        "higher_average_volatility",
        "higher_average_beta",
        "worse_average_drawdown",
    }
    assert result["risk_profile_audit"]["total_monotonicity_violation_count"] == 3





def test_case_37_audit_signature_priority_follows_prompt_order():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    _set_decision_and_report(data, "valuation", "risk_averse", "AAA", "SELL")
    _set_decision_and_report(data, "valuation", "risk_averse", "BBB", "SELL")
    _set_decision_and_report(data, "valuation", "risk_averse", "DDD", "BUY")

    result = _run_valid(fn, data=data)[0]

    assert result["summary"]["all_consensus_decisions_reconciled"] is True
    assert result["summary"]["global_consistency_score"] < 1.0
    assert result["summary"]["audit_signature"] == "portfolio_consistent_with_evidence_warnings"
    assert result["summary"]["audit_signature_code"] == 2

def test_case_36_drawdown_ranking_uses_magnitude_not_raw_negative_value():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()
    portfolio_dates = _business_dates("2024-02-01", "2024-03-15")
    portfolio_index = {day: idx for idx, day in enumerate(portfolio_dates)}

    for row in data["prices"]:
        if row["date"] < "2024-02-01":
            continue

        j = portfolio_index[row["date"]]
        ticker = row["ticker"]

        # AAA, BBB, EEE are monotone rising, so their max drawdown is zero.
        # CCC has a large drawdown; DDD has a still larger drawdown.
        # This guarantees that ordering by raw negative drawdown differs from
        # ordering by drawdown magnitude.
        if ticker == "AAA":
            price = 100.0 + 0.30 * j
        elif ticker == "BBB":
            price = 90.0 + 0.20 * j
        elif ticker == "EEE":
            price = 80.0 + 0.10 * j
        elif ticker == "CCC":
            if j <= 5:
                price = 100.0 + 2.0 * j
            elif j <= 20:
                price = 110.0 - 3.0 * (j - 5)
            else:
                price = 65.0 + 0.15 * (j - 20)
        elif ticker == "DDD":
            if j <= 5:
                price = 100.0 + 2.0 * j
            elif j <= 20:
                price = 110.0 - 5.0 * (j - 5)
            else:
                price = 35.0 + 0.10 * (j - 20)
        else:
            raise AssertionError(f"Unexpected ticker {ticker}")

        for column in ["open", "high", "low", "close", "adjusted_close"]:
            row[column] = f"{price:.12f}"

    result = _run_valid(fn, data=data)[0]
    metrics = result["portfolio_audit"]["portfolio_metrics"]

    expected_magnitude_order = _rank_order(
        {key: abs(value["max_drawdown"]) for key, value in metrics.items()},
        False,
    )

    raw_negative_order = _rank_order(
        {key: value["max_drawdown"] for key, value in metrics.items()},
        False,
    )

    assert raw_negative_order != expected_magnitude_order
    assert result["portfolio_audit"]["rankings"]["by_drawdown_asc"] == expected_magnitude_order


def test_case_38_risk_free_series_name_is_read_from_config_not_hardcoded_rate():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        kwargs = _write_dataset(root, data)
        kwargs["tie_tol"] = 1e-12

        # Change config after writing, so _write_dataset can still use its
        # canonical risk-free writer.
        config = json.loads(Path(kwargs["config_path"]).read_text(encoding="utf-8"))
        config["risk_free_series"] = "ONE_MONTH_RATE"
        Path(kwargs["config_path"]).write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        risk_free_frame = pd.read_csv(kwargs["risk_free_path"], dtype=str)
        risk_free_frame = risk_free_frame.rename(columns={"DGS1MO": "ONE_MONTH_RATE"})
        risk_free_frame.to_csv(kwargs["risk_free_path"], index=False)

        result = _call_valid(fn, **kwargs)
        saved = json.loads(Path(kwargs["output_path"]).read_text(encoding="utf-8"))

        assert saved == result
        assert result["dataset_profile"]["portfolio_trading_day_count"] == (
            len(_business_dates("2024-02-01", "2024-03-15")) - 1
        )
        assert result["portfolio_audit"]["benchmark_metrics"]["sharpe_ratio"] is not None


def test_case_39_exactly_one_terminal_manager_message_is_required():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    original = next(
        row
        for row in data["debate_logs"]
        if row["message_id"] == "AAA_risk_neutral_DEBATE_r3_manager"
    )

    duplicate = copy.deepcopy(original)
    duplicate["message_id"] = "AAA_risk_neutral_DEBATE_r4_manager_duplicate"
    duplicate["round"] = 4
    duplicate["reply_to"] = [original["message_id"]]

    data["debate_logs"].append(duplicate)

    result = _run_valid(fn, data=data)[0]

    debate = result["debate_audit"]["debate_level_results"]["AAA_risk_neutral_DEBATE"]

    assert debate["termination_valid"] is False
    assert result["debate_audit"]["invalid_debate_count"] == 1
    assert result["summary"]["all_debates_valid"] is False


def test_case_40_multi_agent_claim_cannot_directly_use_raw_specialist_evidence():
    fn = multi_agent_equity_consistency_audit

    data = _make_valid_dataset()

    claim = next(
        row
        for row in data["claim_evidence_links"]
        if row["agent_type"] == "multi_agent"
    )

    claim["evidence_id"] = f"{claim['ticker']}_PRICE_VOLUME_JAN2024"
    claim["evidence_source"] = "price_volume"

    result = _run_valid(fn, data=data)[0]

    assert result["evidence_audit"]["claim_level_results"][claim["claim_id"]]["status"] == (
        "role_access_violation"
    )
    assert result["evidence_audit"]["claim_status_counts"]["role_access_violation"] == 1