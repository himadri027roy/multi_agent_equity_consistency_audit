def multi_agent_equity_consistency_audit(
    universe_path="/workspace/data/universe.csv",
    config_path="/workspace/data/config.json",
    feature_schema_path="/workspace/data/feature_schema.json",
    prices_path="/workspace/data/prices.csv",
    risk_free_path="/workspace/data/risk_free.csv",
    stock_features_path="/workspace/data/stock_features.csv",
    price_evidence_path="/workspace/data/price_evidence.jsonl",
    filings_metadata_path="/workspace/data/filings_metadata.csv",
    filings_chunks_path="/workspace/data/filings_chunks.jsonl",
    news_articles_path="/workspace/data/news_articles.jsonl",
    agent_reports_path="/workspace/data/agent_reports.jsonl",
    claim_evidence_links_path="/workspace/data/claim_evidence_links.csv",
    debate_logs_path="/workspace/data/debate_logs.jsonl",
    agent_decisions_path="/workspace/data/agent_decisions.csv",
    output_path="/workspace/output.json",
    tie_tol=1e-12,
):
    import json
    import math
    import re
    from collections import OrderedDict, defaultdict
    from datetime import datetime
    from functools import cmp_to_key
    from pathlib import Path

    import cvxpy as cp
    import mpmath as mp
    import networkx as nx
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    mp.mp.dps = 80

    AGENTS = ["valuation", "fundamental", "sentiment", "multi_agent"]
    SPECIALIST_AGENTS = ["valuation", "fundamental", "sentiment"]
    RISK_PROFILES = ["risk_neutral", "risk_averse", "risk_seeking"]
    DECISIONS = ["BUY", "SELL", "HOLD"]
    CLAIM_STATUSES = [
        "supported",
        "missing_evidence",
        "role_access_violation",
        "decision_mismatch",
        "unsupported",
    ]
    HOLD_POLICY = "exclude"
    PORTFOLIO_WEIGHTING = "equal_weight"
    TRADING_DAYS_PER_YEAR = 252
    ROLLING_SHARPE_WINDOW = 20

    def fail(message):
        raise ValueError(message)

    def clean_string(value, field_name):
        if not isinstance(value, str):
            fail(f"schema violation: {field_name} must be a string")
        return value.strip()

    def parse_number(value, field_name):
        if isinstance(value, (bool, np.bool_)):
            fail(f"invalid numeric value: {field_name} cannot be boolean")
        try:
            if isinstance(value, str):
                text = value.strip()
                if text == "" or text.lower() in {"true", "false"}:
                    fail(f"invalid numeric value: {field_name}")
                number = mp.mpf(text)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                number = mp.mpf(
                    repr(float(value))
                    if isinstance(value, (float, np.floating))
                    else str(int(value))
                )
            else:
                fail(f"invalid numeric value: {field_name}")
        except ValueError:
            raise
        except Exception:
            fail(f"invalid numeric value: {field_name}")
        if not mp.isfinite(number):
            fail(f"non-finite numeric value: {field_name}")
        return number

    def parse_positive_number(value, field_name):
        number = parse_number(value, field_name)
        if number <= 0:
            fail(f"non-positive adjusted prices or numeric value: {field_name}")
        return number

    def parse_nonnegative_number(value, field_name):
        number = parse_number(value, field_name)
        if number < 0:
            fail(f"negative volumes or numeric value: {field_name}")
        return number

    def parse_int(value, field_name):
        number = parse_number(value, field_name)
        if number != int(number):
            fail(f"invalid integer value: {field_name}")
        return int(number)

    def parse_bool(value, field_name):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "y"}:
                return True
            if text in {"0", "false", "no", "n", ""}:
                return False
        fail(f"invalid boolean value: {field_name}")

    def parse_tie_tol(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            fail("invalid scalar arguments: tie_tol")
        number = mp.mpf(repr(float(value)))
        if not mp.isfinite(number) or number <= 0:
            fail("invalid scalar arguments: tie_tol")
        return number

    tol = parse_tie_tol(tie_tol)

    def greater_than(a, b):
        return a > b + tol

    def less_than(a, b):
        return a < b - tol

    def parse_date(value, field_name):
        if not isinstance(value, str):
            fail(f"invalid date: {field_name}")
        text = value.strip()
        try:
            return pd.Timestamp(datetime.strptime(text, "%Y-%m-%d").date())
        except Exception:
            fail(f"invalid date: {field_name}")

    def round_float(value):
        if value is None:
            return None
        f = float(value)
        if not math.isfinite(f):
            return None
        out = round(f, 12)
        return 0.0 if out == -0.0 else out

    def copied_float(value):
        if value is None:
            return None
        f = float(value)
        if not math.isfinite(f):
            fail("non-finite numeric value: copied float")
        return f

    def mp_mean(values, field_name):
        if not values:
            fail(f"empty portfolios where a metric is required: {field_name}")
        return mp.fsum(values) / mp.mpf(len(values))

    def mp_sample_std(values, field_name):
        if len(values) < 2:
            fail(f"invalid portfolio windows: at least two observations for {field_name}")
        mean_value = mp_mean(values, field_name)
        variance = mp.fsum([(x - mean_value) ** 2 for x in values]) / mp.mpf(
            len(values) - 1
        )
        if variance < 0 and abs(variance) <= tol:
            variance = mp.mpf("0")
        if variance < 0:
            fail(f"non-finite numeric value: negative variance for {field_name}")
        return mp.sqrt(variance)

    def clean_json(value, field_name):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return [clean_json(x, f"{field_name}[]") for x in value]
        if isinstance(value, dict):
            out = {}
            for raw_key, raw_value in value.items():
                key = clean_string(raw_key, f"{field_name} key")
                if key in out:
                    fail(
                        f"duplicate primary identifiers: duplicate cleaned key in {field_name}"
                    )
                out[key] = clean_json(raw_value, f"{field_name}.{key}")
            return out
        return value

    def load_json(path, top_level, name):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            fail(f"malformed JSON content in {name}: {exc}")
        except OSError as exc:
            fail(f"read failure in {name}: {exc}")
        payload = clean_json(payload, name)
        if top_level == "object" and not isinstance(payload, dict):
            fail(f"schema violation: {name} must contain an object")
        if top_level == "array" and not isinstance(payload, list):
            fail(f"schema violation: {name} must contain an array")
        return payload

    def load_jsonl(path, name):
        records = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.strip() == "":
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        fail(
                            f"malformed JSONL content in {name} line {line_number}: {exc}"
                        )
                    record = clean_json(record, f"{name}[{line_number}]")
                    if not isinstance(record, dict):
                        fail(
                            f"schema violation: {name} line {line_number} must be an object"
                        )
                    records.append(record)
        except OSError as exc:
            fail(f"read failure in {name}: {exc}")
        return records

    def load_csv(path, required_columns, name):
        try:
            frame = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8",
            )
        except Exception as exc:
            fail(f"malformed CSV content in {name}: {exc}")
        frame.columns = [str(col).strip() for col in frame.columns]
        if len(set(frame.columns)) != len(frame.columns):
            fail(f"duplicate primary identifiers: duplicate CSV column in {name}")
        missing = [col for col in required_columns if col not in frame.columns]
        if missing:
            fail(f"missing required columns in {name}: {missing}")
        for col in frame.columns:
            frame[col] = frame[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        return frame

    def require_unique(frame, columns, name):
        if bool(frame.duplicated(columns, keep=False).any()):
            fail(f"duplicate primary identifiers in {name}: {columns}")

    config = load_json(config_path, "object", "config.json")
    feature_schema = load_json(feature_schema_path, "object", "feature_schema.json")

    required_config = [
        "analysis_start",
        "analysis_end",
        "portfolio_start",
        "portfolio_end",
        "trading_days_per_year",
        "rolling_sharpe_window",
        "risk_free_series",
        "portfolio_weighting",
        "allowed_agents",
        "risk_profiles",
        "allowed_decisions",
        "hold_policy",
        "minimum_turns_per_agent",
        "termination_token",
    ]
    for key in required_config:
        if key not in config:
            fail(f"missing required keys in config.json: {key}")

    analysis_start = parse_date(config["analysis_start"], "config.analysis_start")
    analysis_end = parse_date(config["analysis_end"], "config.analysis_end")
    portfolio_start = parse_date(config["portfolio_start"], "config.portfolio_start")
    portfolio_end = parse_date(config["portfolio_end"], "config.portfolio_end")
    if analysis_end < analysis_start or portfolio_end < portfolio_start or not (
        portfolio_start > analysis_end
    ):
        fail("invalid portfolio windows")

    trading_days = parse_int(
        config["trading_days_per_year"], "config.trading_days_per_year"
    )
    rolling_window = parse_int(
        config["rolling_sharpe_window"], "config.rolling_sharpe_window"
    )
    min_turns = parse_int(
        config["minimum_turns_per_agent"], "config.minimum_turns_per_agent"
    )
    risk_free_series_name = clean_string(
        config["risk_free_series"], "config.risk_free_series"
    )
    termination_token = clean_string(
        config["termination_token"], "config.termination_token"
    )

    if trading_days != TRADING_DAYS_PER_YEAR:
        fail("contradiction with fixed benchmark contract: trading_days_per_year")
    if rolling_window != ROLLING_SHARPE_WINDOW or rolling_window <= 1:
        fail("contradiction with fixed benchmark contract: rolling_sharpe_window")
    if min_turns <= 0 or termination_token == "":
        fail("invalid debate references: minimum turns or termination token")
    if clean_string(config["portfolio_weighting"], "config.portfolio_weighting") != PORTFOLIO_WEIGHTING:
        fail("contradiction with fixed benchmark contract: portfolio_weighting")
    if clean_string(config["hold_policy"], "config.hold_policy") != HOLD_POLICY:
        fail("contradiction with fixed benchmark contract: hold_policy")
    if (
        config["allowed_agents"] != AGENTS
        or config["risk_profiles"] != RISK_PROFILES
        or config["allowed_decisions"] != DECISIONS
    ):
        fail("contradiction with fixed benchmark contract: canonical axes")
    if set(feature_schema.keys()) != set(AGENTS):
        fail("missing required keys in feature_schema.json")
    for role, features in feature_schema.items():
        if role not in AGENTS or not isinstance(features, list):
            fail("invalid labels in feature_schema.json")
        for item in features:
            clean_string(item, f"feature_schema.{role}[]")

    universe_df = load_csv(
        universe_path,
        ["ticker", "company_name", "sector", "industry", "benchmark_weight"],
        "universe.csv",
    )
    prices_df = load_csv(
        prices_path,
        ["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"],
        "prices.csv",
    )
    risk_free_df = load_csv(risk_free_path, ["date"], "risk_free.csv")
    stock_features_df = load_csv(
        stock_features_path,
        [
            "ticker",
            "january_return",
            "january_annualized_volatility",
            "january_max_drawdown",
            "january_beta_to_equal_weight_universe",
            "january_average_volume",
        ],
        "stock_features.csv",
    )
    filings_metadata_df = load_csv(
        filings_metadata_path,
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
        "filings_metadata.csv",
    )
    claims_df = load_csv(
        claim_evidence_links_path,
        [
            "claim_id",
            "report_id",
            "ticker",
            "agent_type",
            "claim_type",
            "claim_text",
            "evidence_id",
            "evidence_source",
        ],
        "claim_evidence_links.csv",
    )
    decisions_df = load_csv(
        agent_decisions_path,
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
        "agent_decisions.csv",
    )

    require_unique(universe_df, ["ticker"], "universe.csv")
    tickers = list(universe_df["ticker"])
    ticker_set = set(tickers)
    if not tickers or any(t == "" for t in tickers):
        fail("inconsistent ticker coverage: empty universe or ticker")
    for idx, row in universe_df.iterrows():
        parse_nonnegative_number(
            row["benchmark_weight"], f"universe.benchmark_weight[{idx}]"
        )

    require_unique(stock_features_df, ["ticker"], "stock_features.csv")
    if set(stock_features_df["ticker"]) != ticker_set:
        fail("inconsistent ticker coverage: stock_features.csv")
    feature_columns = [
        "january_return",
        "january_annualized_volatility",
        "january_max_drawdown",
        "january_beta_to_equal_weight_universe",
        "january_average_volume",
    ]
    feature_map = {}
    for idx, row in stock_features_df.iterrows():
        ticker = clean_string(row["ticker"], f"stock_features.ticker[{idx}]")
        values = {
            col: parse_number(row[col], f"stock_features.{col}[{idx}]")
            for col in feature_columns
        }
        if values["january_annualized_volatility"] < 0:
            fail("non-finite numeric value: negative January volatility")
        if values["january_average_volume"] < 0:
            fail("negative volumes: stock_features.january_average_volume")
        feature_map[ticker] = values

    require_unique(prices_df, ["date", "ticker"], "prices.csv")
    price_records = []
    price_tickers = set()
    for idx, row in prices_df.iterrows():
        date = parse_date(row["date"], f"prices.date[{idx}]")
        ticker = clean_string(row["ticker"], f"prices.ticker[{idx}]")
        if ticker not in ticker_set:
            fail("inconsistent ticker coverage: prices.csv")
        price_tickers.add(ticker)
        record = {"date": date, "ticker": ticker}
        for col in ["open", "high", "low", "close", "adjusted_close"]:
            record[col] = parse_positive_number(row[col], f"prices.{col}[{idx}]")
        record["volume"] = parse_nonnegative_number(
            row["volume"], f"prices.volume[{idx}]"
        )
        price_records.append(record)
    if price_tickers != ticker_set:
        fail("inconsistent ticker coverage: prices.csv missing ticker")

    require_unique(filings_metadata_df, ["accession_number"], "filings_metadata.csv")
    for idx, row in filings_metadata_df.iterrows():
        ticker = clean_string(row["ticker"], f"filings_metadata.ticker[{idx}]")
        if ticker not in ticker_set:
            fail("inconsistent ticker coverage: filings_metadata.csv")
        if clean_string(row["form_type"], f"filings_metadata.form_type[{idx}]") not in {
            "10-K",
            "10-Q",
        }:
            fail("invalid labels: filings_metadata.form_type")
        parse_int(row["cik"], f"filings_metadata.cik[{idx}]")
        parse_date(row["filing_date"], f"filings_metadata.filing_date[{idx}]")
        parse_date(row["period_end"], f"filings_metadata.period_end[{idx}]")
        if clean_string(
            row["accession_number"], f"filings_metadata.accession_number[{idx}]"
        ) == "":
            fail("missing required keys: filings_metadata.accession_number")

    require_unique(
        decisions_df,
        ["ticker", "agent_type", "risk_profile"],
        "agent_decisions.csv",
    )
    decision_table = {}
    expected_decision_keys = {
        (ticker, agent, risk)
        for ticker in tickers
        for agent in AGENTS
        for risk in RISK_PROFILES
    }
    for idx, row in decisions_df.iterrows():
        date = parse_date(row["date"], f"agent_decisions.date[{idx}]")
        ticker = clean_string(row["ticker"], f"agent_decisions.ticker[{idx}]")
        agent = clean_string(row["agent_type"], f"agent_decisions.agent_type[{idx}]")
        risk = clean_string(row["risk_profile"], f"agent_decisions.risk_profile[{idx}]")
        decision = clean_string(row["decision"], f"agent_decisions.decision[{idx}]")
        confidence = parse_number(row["confidence"], f"agent_decisions.confidence[{idx}]")
        if ticker not in ticker_set or agent not in AGENTS or risk not in RISK_PROFILES:
            fail("invalid labels in agent_decisions.csv")
        if decision not in DECISIONS:
            fail("invalid decisions in agent_decisions.csv")
        if confidence < 0 or confidence > 1:
            fail("invalid numeric value: decision confidence must be in [0,1]")
        decision_table[(ticker, agent, risk)] = {
            "date": date,
            "decision": decision,
            "confidence": confidence,
            "evidence_ids": clean_string(
                row["evidence_ids"], f"agent_decisions.evidence_ids[{idx}]"
            ),
        }
    if set(decision_table.keys()) != expected_decision_keys:
        fail("incomplete required specialist coverage: agent_decisions.csv")

    price_evidence = load_jsonl(price_evidence_path, "price_evidence.jsonl")
    filings_chunks = load_jsonl(filings_chunks_path, "filings_chunks.jsonl")
    news_articles = load_jsonl(news_articles_path, "news_articles.jsonl")
    agent_reports = load_jsonl(agent_reports_path, "agent_reports.jsonl")
    debate_logs = load_jsonl(debate_logs_path, "debate_logs.jsonl")

    evidence_map = OrderedDict()

    def add_evidence(
        evidence_id,
        source,
        channel,
        ticker,
        payload,
        evidence_name,
        risk_profile=None,
    ):
        evidence_id = clean_string(evidence_id, f"{evidence_name}.evidence_id")
        source = clean_string(source, f"{evidence_name}.source")
        ticker = clean_string(ticker, f"{evidence_name}.ticker")
        if evidence_id == "":
            fail(f"invalid evidence references: empty id in {evidence_name}")
        if ticker not in ticker_set:
            fail(f"inconsistent ticker coverage: {evidence_name}")
        if evidence_id in evidence_map:
            fail(f"duplicate primary identifiers: evidence id {evidence_id}")
        evidence_map[evidence_id] = {
            "source": source,
            "channel": channel,
            "ticker": ticker,
            "risk_profile": risk_profile,
            "payload": payload,
        }

    for idx, record in enumerate(price_evidence):
        for key in ["evidence_id", "ticker", "source_type", "start_date", "end_date"]:
            if key not in record:
                fail(f"missing required keys in price_evidence.jsonl: {key}")
        start = parse_date(record["start_date"], f"price_evidence.start_date[{idx}]")
        end = parse_date(record["end_date"], f"price_evidence.end_date[{idx}]")
        if start < analysis_start or end > analysis_end or end < start:
            fail("invalid evidence references: price evidence outside analysis window")
        source = clean_string(
            record["source_type"], f"price_evidence.source_type[{idx}]"
        )
        if source != "price_volume":
            fail("invalid labels: price evidence source_type")
        add_evidence(
            record["evidence_id"],
            source,
            "valuation",
            record["ticker"],
            record,
            "price_evidence",
        )

    for idx, record in enumerate(filings_chunks):
        for key in ["chunk_id", "ticker", "source_type", "as_of_date"]:
            if key not in record:
                fail(f"missing required keys in filings_chunks.jsonl: {key}")
        as_of = parse_date(record["as_of_date"], f"filings_chunks.as_of_date[{idx}]")
        if as_of > analysis_end:
            fail("invalid evidence references: filing chunk outside analysis window")
        source = clean_string(
            record["source_type"], f"filings_chunks.source_type[{idx}]"
        )
        add_evidence(
            record["chunk_id"],
            source,
            "fundamental",
            record["ticker"],
            record,
            "filings_chunks",
        )

    for idx, record in enumerate(news_articles):
        for key in ["news_id", "ticker", "date", "title", "body", "url"]:
            if key not in record:
                fail(f"missing required keys in news_articles.jsonl: {key}")
        raw_date = clean_string(record["date"], f"news_articles.date[{idx}]")
        if raw_date:
            news_date = parse_date(raw_date, f"news_articles.date[{idx}]")
            if news_date < analysis_start or news_date > portfolio_start:
                fail("invalid evidence references: news outside allowed pre-decision window")
        elif "NO_PUBLIC_NEWS" not in clean_string(
            record["news_id"], f"news_articles.news_id[{idx}]"
        ):
            fail(f"invalid date: news_articles.date[{idx}]")
        source_name = clean_string(
            record.get("source", "news"), f"news_articles.source[{idx}]"
        )
        source = (
            "GDELT_NEWS"
            if source_name.upper() == "GDELT"
            else (f"{source_name}_NEWS" if source_name else "news_article")
        )
        add_evidence(
            record["news_id"],
            source,
            "sentiment",
            record["ticker"],
            record,
            "news_articles",
        )

    reports_map = OrderedDict()
    report_claim_ids = {}
    for idx, record in enumerate(agent_reports):
        for key in [
            "report_id",
            "ticker",
            "agent_type",
            "risk_profile",
            "final_recommendation",
            "confidence",
            "claim_ids",
        ]:
            if key not in record:
                fail(f"missing required keys in agent_reports.jsonl: {key}")
        report_id = clean_string(record["report_id"], f"agent_reports.report_id[{idx}]")
        if report_id == "" or report_id in reports_map:
            fail("duplicate primary identifiers: report_id")
        ticker = clean_string(record["ticker"], f"agent_reports.ticker[{idx}]")
        agent = clean_string(record["agent_type"], f"agent_reports.agent_type[{idx}]")
        risk = clean_string(record["risk_profile"], f"agent_reports.risk_profile[{idx}]")
        recommendation = clean_string(
            record["final_recommendation"],
            f"agent_reports.final_recommendation[{idx}]",
        )
        confidence = parse_number(
            record["confidence"], f"agent_reports.confidence[{idx}]"
        )
        if ticker not in ticker_set or agent not in AGENTS or risk not in RISK_PROFILES:
            fail("invalid labels in agent_reports.jsonl")
        if recommendation not in DECISIONS:
            fail("invalid decisions in agent_reports.jsonl")
        if confidence < 0 or confidence > 1:
            fail("invalid numeric value: report confidence must be in [0,1]")
        if not isinstance(record["claim_ids"], list):
            fail("schema violation: agent_reports.claim_ids must be an array")
        claim_ids = []
        seen_claims = set()
        for claim_id in record["claim_ids"]:
            clean_claim = clean_string(claim_id, f"agent_reports.claim_ids[{idx}]")
            if clean_claim == "" or clean_claim in seen_claims:
                fail("invalid claim references: empty or duplicate report claim_id")
            seen_claims.add(clean_claim)
            claim_ids.append(clean_claim)
        reports_map[report_id] = {
            "ticker": ticker,
            "agent": agent,
            "risk_profile": risk,
            "final_recommendation": recommendation,
            "confidence": confidence,
            "claim_ids": claim_ids,
        }
        report_claim_ids[report_id] = set(claim_ids)
        source = f"{agent}_report" if agent in SPECIALIST_AGENTS else "consensus_decision"
        add_evidence(
            report_id,
            source,
            "multi_agent",
            ticker,
            record,
            "agent_reports",
            risk_profile=risk,
        )

    expected_report_keys = {
        (ticker, agent, risk)
        for ticker in tickers
        for agent in AGENTS
        for risk in RISK_PROFILES
    }
    observed_report_keys = {
        (item["ticker"], item["agent"], item["risk_profile"])
        for item in reports_map.values()
    }
    if observed_report_keys != expected_report_keys:
        fail("incomplete required specialist coverage: agent_reports.jsonl")

    debate_records_by_id = defaultdict(list)
    seen_message_ids = set()
    for idx, record in enumerate(debate_logs):
        for key in [
            "debate_id",
            "message_id",
            "ticker",
            "risk_profile",
            "round",
            "speaker",
            "message",
            "explicit_decision",
            "confidence",
            "reply_to",
        ]:
            if key not in record:
                fail(f"missing required keys in debate_logs.jsonl: {key}")
        debate_id = clean_string(record["debate_id"], f"debate_logs.debate_id[{idx}]")
        message_id = clean_string(record["message_id"], f"debate_logs.message_id[{idx}]")
        if debate_id == "" or message_id == "" or message_id in seen_message_ids:
            fail("duplicate primary identifiers or empty debate/message id")
        seen_message_ids.add(message_id)
        ticker = clean_string(record["ticker"], f"debate_logs.ticker[{idx}]")
        risk = clean_string(record["risk_profile"], f"debate_logs.risk_profile[{idx}]")
        if ticker not in ticker_set or risk not in RISK_PROFILES:
            fail("invalid labels in debate_logs.jsonl")
        record["round"] = parse_int(record["round"], f"debate_logs.round[{idx}]")
        if record["round"] <= 0:
            fail("malformed graph edges: non-positive round")
        decision = clean_string(
            record["explicit_decision"], f"debate_logs.explicit_decision[{idx}]"
        )
        if decision not in DECISIONS:
            fail("invalid decisions in debate_logs.jsonl")
        record["explicit_decision"] = decision
        record["confidence"] = parse_number(
            record["confidence"], f"debate_logs.confidence[{idx}]"
        )
        if record["confidence"] < 0 or record["confidence"] > 1:
            fail("invalid numeric value: debate confidence must be in [0,1]")
        if not isinstance(record["reply_to"], list):
            fail("malformed graph edges: reply_to must be an array")
        record["reply_to"] = [
            clean_string(parent, f"debate_logs.reply_to[{idx}]")
            for parent in record["reply_to"]
        ]
        if "evidence_ids" not in record:
            record["evidence_ids"] = []
        if not isinstance(record["evidence_ids"], list):
            fail("invalid evidence references: debate evidence_ids must be an array")
        record["evidence_ids"] = [
            clean_string(evid, f"debate_logs.evidence_ids[{idx}]")
            for evid in record["evidence_ids"]
        ]
        debate_records_by_id[debate_id].append(record)

    expected_debate_keys = {(ticker, risk) for ticker in tickers for risk in RISK_PROFILES}
    observed_debate_keys = set()
    for debate_id, rows in debate_records_by_id.items():
        ticker_values = {row["ticker"] for row in rows}
        risk_values = {row["risk_profile"] for row in rows}
        if len(ticker_values) != 1 or len(risk_values) != 1:
            fail("invalid debate references: debate mixes ticker or risk profile")
        ticker = next(iter(ticker_values))
        risk = next(iter(risk_values))
        observed_debate_keys.add((ticker, risk))
        add_evidence(
            debate_id,
            "debate_log",
            "multi_agent",
            ticker,
            {"messages": rows},
            "debate_logs",
            risk_profile=risk,
        )
    if observed_debate_keys != expected_debate_keys:
        fail("incomplete required specialist coverage: debates")
    for rows in debate_records_by_id.values():
        for row in rows:
            for evidence_id in row["evidence_ids"]:
                if evidence_id not in evidence_map:
                    fail("invalid evidence references: debate message evidence_ids")

    def split_evidence_ids(text):
        text = clean_string(text, "evidence_ids")
        if text == "":
            return []
        return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]

    for info in decision_table.values():
        for evidence_id in split_evidence_ids(info["evidence_ids"]):
            if evidence_id not in evidence_map:
                fail("invalid evidence references: agent_decisions.evidence_ids")

    require_unique(claims_df, ["claim_id"], "claim_evidence_links.csv")
    allowed_channel_by_agent = {
        "valuation": "valuation",
        "fundamental": "fundamental",
        "sentiment": "sentiment",
        "multi_agent": "multi_agent",
    }
    claim_status_counts = {status: 0 for status in CLAIM_STATUSES}
    agent_claim_stats = {
        agent: {"claim_count": 0, "supported_count": 0, "violation_count": 0}
        for agent in AGENTS
    }
    claim_level_results = OrderedDict()
    observed_report_claim_ids = defaultdict(set)

    for idx, row in claims_df.iterrows():
        claim_id = clean_string(row["claim_id"], f"claims.claim_id[{idx}]")
        report_id = clean_string(row["report_id"], f"claims.report_id[{idx}]")
        ticker = clean_string(row["ticker"], f"claims.ticker[{idx}]")
        agent = clean_string(row["agent_type"], f"claims.agent_type[{idx}]")
        evidence_id = clean_string(row["evidence_id"], f"claims.evidence_id[{idx}]")
        row_evidence_source = clean_string(
            row["evidence_source"], f"claims.evidence_source[{idx}]"
        )
        if report_id not in reports_map:
            fail("invalid claim references: missing report_id")
        report = reports_map[report_id]
        if (
            ticker != report["ticker"]
            or agent != report["agent"]
            or claim_id not in report_claim_ids[report_id]
        ):
            fail("invalid claim references: claim/report mismatch")
        if ticker not in ticker_set or agent not in AGENTS:
            fail("invalid labels in claim_evidence_links.csv")
        observed_report_claim_ids[report_id].add(claim_id)
        risk = report["risk_profile"]
        report_decision = report["final_recommendation"]
        table_decision = decision_table[(ticker, agent, risk)]["decision"]
        expected_status = (
            clean_string(row["expected_status"], f"claims.expected_status[{idx}]")
            if "expected_status" in claims_df.columns
            else ""
        )
        allow_missing = expected_status == "missing_evidence"
        if "allow_missing_evidence" in claims_df.columns:
            allow_missing = allow_missing or parse_bool(
                row["allow_missing_evidence"],
                f"claims.allow_missing_evidence[{idx}]",
            )

        actual_source = row_evidence_source
        if report_decision != table_decision:
            status = "decision_mismatch"
        elif evidence_id not in evidence_map:
            if allow_missing:
                status = "missing_evidence"
            else:
                fail("invalid evidence references: claim evidence_id")
        else:
            evidence = evidence_map[evidence_id]
            actual_source = evidence["source"]
            if row_evidence_source and row_evidence_source != actual_source:
                status = "unsupported"
            elif evidence["ticker"] != ticker:
                status = "unsupported"
            elif evidence["risk_profile"] is not None and evidence["risk_profile"] != risk:
                status = "unsupported"
            elif evidence["channel"] != allowed_channel_by_agent[agent]:
                status = "role_access_violation"
            else:
                status = "supported"

        claim_status_counts[status] += 1
        agent_claim_stats[agent]["claim_count"] += 1
        if status == "supported":
            agent_claim_stats[agent]["supported_count"] += 1
        if status == "role_access_violation":
            agent_claim_stats[agent]["violation_count"] += 1
        claim_level_results[claim_id] = {
            "ticker": ticker,
            "agent": agent,
            "risk_profile": risk,
            "status": status,
            "evidence_id": evidence_id,
            "evidence_source": actual_source,
        }

    for report_id, declared_claim_ids in report_claim_ids.items():
        if observed_report_claim_ids[report_id] != declared_claim_ids:
            fail(
                "invalid claim references: report claim_ids and claim_evidence_links coverage disagree"
            )

    agent_claim_scores = OrderedDict()
    for agent in AGENTS:
        stats = agent_claim_stats[agent]
        support_rate = (
            None
            if stats["claim_count"] == 0
            else mp.mpf(stats["supported_count"]) / mp.mpf(stats["claim_count"])
        )
        agent_claim_scores[agent] = {
            "claim_count": int(stats["claim_count"]),
            "supported_count": int(stats["supported_count"]),
            "support_rate": round_float(support_rate),
            "violation_count": int(stats["violation_count"]),
        }

    speaker_order = [
        "valuation_agent",
        "fundamental_agent",
        "sentiment_agent",
        "group_chat_manager",
    ]
    specialist_speaker = {
        "valuation": "valuation_agent",
        "fundamental": "fundamental_agent",
        "sentiment": "sentiment_agent",
    }
    debate_level_results = OrderedDict()
    invalid_debate_count = 0
    unresolved_dissent_count = 0
    valid_and_reconciled_count = 0

    def debate_sort_key(debate_id):
        item = evidence_map[debate_id]
        return (
            tickers.index(item["ticker"]),
            RISK_PROFILES.index(item["risk_profile"]),
            debate_id,
        )

    for debate_id in sorted(debate_records_by_id.keys(), key=debate_sort_key):
        rows = list(debate_records_by_id[debate_id])
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r["round"],
                speaker_order.index(r["speaker"])
                if r["speaker"] in speaker_order
                else len(speaker_order),
                r["message_id"],
            ),
        )
        ticker = rows_sorted[0]["ticker"]
        risk = rows_sorted[0]["risk_profile"]
        message_map = {row["message_id"]: row for row in rows_sorted}
        if len(message_map) != len(rows_sorted):
            fail("duplicate primary identifiers: message_id within debate")
        graph = nx.DiGraph()
        for row in rows_sorted:
            graph.add_node(row["message_id"], speaker=row["speaker"], round=row["round"])
        for row in rows_sorted:
            for parent in row["reply_to"]:
                if parent not in message_map:
                    fail("malformed graph edges: reply_to missing parent")
                if (
                    message_map[parent]["debate_id"] != debate_id
                    or message_map[parent]["round"] >= row["round"]
                ):
                    fail("malformed graph edges: invalid reply_to ordering")
                graph.add_edge(parent, row["message_id"])
        if not nx.is_directed_acyclic_graph(graph):
            fail("malformed graph edges: debate graph is cyclic")

        turn_counts = OrderedDict()
        for agent in SPECIALIST_AGENTS:
            turn_counts[agent] = sum(
                1 for row in rows_sorted if row["speaker"] == specialist_speaker[agent]
            )
        if any(turn_counts[agent] < min_turns for agent in SPECIALIST_AGENTS):
            fail("incomplete required specialist coverage: debate turns")

        terminal_candidates = [
            row
            for row in rows_sorted
            if row["speaker"] == "group_chat_manager"
            and termination_token in clean_string(row["message"], "debate.message")
        ]
        terminal_valid = len(terminal_candidates) == 1
        terminal_message = terminal_candidates[0] if terminal_valid else None
        terminal_decision = (
            terminal_message["explicit_decision"] if terminal_message is not None else ""
        )
        terminal_confidence = (
            terminal_message["confidence"] if terminal_message is not None else None
        )
        if terminal_message is not None:
            terminal_valid = (
                terminal_valid
                and graph.out_degree(terminal_message["message_id"]) == 0
                and terminal_decision in DECISIONS
                and terminal_message["round"] == max(row["round"] for row in rows_sorted)
            )
        consensus_match = bool(
            terminal_decision == decision_table[(ticker, "multi_agent", risk)]["decision"]
        )

        final_specialist_decisions = []
        for agent in SPECIALIST_AGENTS:
            rows_for_speaker = [
                row
                for row in rows_sorted
                if row["speaker"] == specialist_speaker[agent]
                and row["explicit_decision"] in DECISIONS
            ]
            if not rows_for_speaker:
                fail(
                    "incomplete required specialist coverage: missing specialist explicit decision"
                )
            final_specialist_decisions.append(
                sorted(rows_for_speaker, key=lambda r: (r["round"], r["message_id"]))[-1][
                    "explicit_decision"
                ]
            )
        unresolved_dissent = terminal_decision in DECISIONS and sum(
            1 for decision in final_specialist_decisions if decision != terminal_decision
        ) >= 2

        try:
            pagerank = nx.pagerank(graph)
        except Exception as exc:
            fail(f"malformed graph edges: PageRank failure: {exc}")
        speaker_scores_raw = defaultdict(lambda: mp.mpf("0"))
        for node, score in pagerank.items():
            speaker_scores_raw[graph.nodes[node]["speaker"]] += mp.mpf(
                repr(float(score))
            )
        speakers = [speaker for speaker in speaker_order if speaker in speaker_scores_raw]
        speakers.extend(
            sorted(speaker for speaker in speaker_scores_raw if speaker not in speaker_order)
        )
        speaker_pagerank = OrderedDict(
            (speaker, round_float(speaker_scores_raw[speaker])) for speaker in speakers
        )

        if terminal_valid and consensus_match:
            valid_and_reconciled_count += 1
        else:
            invalid_debate_count += 1
        if unresolved_dissent:
            unresolved_dissent_count += 1
        debate_level_results[debate_id] = {
            "ticker": ticker,
            "risk_profile": risk,
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
            "weak_component_count": int(nx.number_weakly_connected_components(graph)),
            "turn_count_by_specialist": turn_counts,
            "terminal_decision": terminal_decision,
            "terminal_confidence": copied_float(terminal_confidence),
            "termination_valid": bool(terminal_valid),
            "consensus_matches_decision_table": bool(consensus_match),
            "unresolved_dissent": bool(unresolved_dissent),
            "speaker_pagerank": speaker_pagerank,
        }

    debate_count = len(debate_records_by_id)
    global_debate_score = (
        mp.mpf("1")
        if debate_count == 0
        else mp.mpf(valid_and_reconciled_count) / mp.mpf(debate_count)
    )

    all_price_dates = sorted({record["date"] for record in price_records})
    portfolio_price_dates = [
        date for date in all_price_dates if portfolio_start <= date <= portfolio_end
    ]
    if len(portfolio_price_dates) < 2:
        fail("invalid portfolio windows: fewer than two portfolio price dates")
    price_by_ticker_date = {
        (record["ticker"], record["date"]): record for record in price_records
    }
    for ticker in tickers:
        for date in portfolio_price_dates:
            if (ticker, date) not in price_by_ticker_date:
                fail(
                    "inconsistent ticker coverage: missing portfolio adjusted-close observation"
                )
    return_dates = portfolio_price_dates[1:]
    ticker_returns = {ticker: [] for ticker in tickers}
    for ticker in tickers:
        for previous_date, current_date in zip(
            portfolio_price_dates[:-1], portfolio_price_dates[1:]
        ):
            previous_price = price_by_ticker_date[(ticker, previous_date)][
                "adjusted_close"
            ]
            current_price = price_by_ticker_date[(ticker, current_date)][
                "adjusted_close"
            ]
            ticker_returns[ticker].append(current_price / previous_price - 1)

    require_unique(risk_free_df, ["date"], "risk_free.csv")
    rf_columns = [col for col in risk_free_df.columns if col != "date"]
    if risk_free_series_name in risk_free_df.columns:
        rf_column = risk_free_series_name
    elif len(rf_columns) == 1:
        rf_column = rf_columns[0]
    else:
        fail(
            "risk-free series that cannot be filled over the portfolio window: ambiguous column"
        )
    rf_records = []
    for idx, row in risk_free_df.iterrows():
        date = parse_date(row["date"], f"risk_free.date[{idx}]")
        annualized_percent = parse_number(row[rf_column], f"risk_free.{rf_column}[{idx}]")
        rf_records.append(
            (date, annualized_percent / (mp.mpf("100") * mp.mpf(trading_days)))
        )
    rf_series = pd.Series({date: rate for date, rate in rf_records}, dtype=object).sort_index()
    if rf_series.empty:
        fail("risk-free series that cannot be filled over the portfolio window: empty")
    rf_aligned = rf_series.reindex(portfolio_price_dates).ffill().bfill()
    if rf_aligned.isna().any():
        fail("risk-free series that cannot be filled over the portfolio window")
    risk_free_returns = [rf_aligned.loc[date] for date in return_dates]

    def equal_weight_returns(selected_tickers, field_name):
        if not selected_tickers:
            fail(f"empty portfolios where a metric is required: {field_name}")
        denom = mp.mpf(len(selected_tickers))
        return [
            mp.fsum(ticker_returns[ticker][idx] for ticker in selected_tickers) / denom
            for idx in range(len(return_dates))
        ]

    def compound_return(returns):
        product = mp.mpf("1")
        for value in returns:
            product *= 1 + value
        return product - 1

    def max_drawdown(returns):
        value = mp.mpf("1")
        running_max = mp.mpf("1")
        worst = mp.mpf("0")
        for ret in returns:
            value *= 1 + ret
            if value > running_max:
                running_max = value
            drawdown = value / running_max - 1
            if drawdown < worst:
                worst = drawdown
        return worst

    def sharpe_ratio(returns, rf_returns, field_name):
        excess = [ret - rf for ret, rf in zip(returns, rf_returns)]
        std = mp_sample_std(excess, f"{field_name} excess returns")
        if abs(std) <= tol:
            return None
        return mp_mean(excess, f"{field_name} excess mean") / std * mp.sqrt(
            mp.mpf(trading_days)
        )

    def rolling_sharpe_values(returns, rf_returns, field_name):
        if len(returns) < rolling_window:
            fail("invalid portfolio windows: insufficient observations for rolling Sharpe")
        values = []
        for end in range(rolling_window, len(returns) + 1):
            excess = [
                ret - rf
                for ret, rf in zip(
                    returns[end - rolling_window:end],
                    rf_returns[end - rolling_window:end],
                )
            ]
            std = mp_sample_std(excess, f"{field_name} rolling excess returns")
            values.append(
                None
                if abs(std) <= tol
                else mp_mean(excess, f"{field_name} rolling excess mean")
                / std
                * mp.sqrt(mp.mpf(trading_days))
            )
        return values

    benchmark_tickers = [
        ticker
        for ticker in tickers
        if all((ticker, date) in price_by_ticker_date for date in portfolio_price_dates)
    ]
    if not benchmark_tickers:
        fail("empty portfolios where a metric is required: benchmark")
    benchmark_returns = equal_weight_returns(benchmark_tickers, "benchmark")
    benchmark_cumulative = compound_return(benchmark_returns)
    benchmark_volatility = mp.sqrt(mp.mpf(trading_days)) * mp_sample_std(
        benchmark_returns, "benchmark returns"
    )
    benchmark_drawdown = max_drawdown(benchmark_returns)
    benchmark_sharpe = sharpe_ratio(benchmark_returns, risk_free_returns, "benchmark")

    def ols_diagnostics(portfolio_returns, benchmark_returns, rf_returns, field_name):
        y = np.array(
            [float(ret - rf) for ret, rf in zip(portfolio_returns, rf_returns)],
            dtype=float,
        )
        x = np.array(
            [float(ret - rf) for ret, rf in zip(benchmark_returns, rf_returns)],
            dtype=float,
        )
        if len(y) != len(x) or len(y) < 3:
            fail(
                f"invalid portfolio windows: OLS requires at least three observations for {field_name}"
            )
        X = sm.add_constant(x, has_constant="add")
        try:
            fit = sm.OLS(y, X).fit()
        except Exception as exc:
            fail(f"OLS diagnostic failure for {field_name}: {exc}")
        params = np.asarray(fit.params, dtype=float)
        if params.shape[0] != 2 or not np.all(np.isfinite(params)):
            fail(f"non-finite numeric value: OLS parameters for {field_name}")
        alpha = mp.mpf(repr(float(params[0])))
        beta = mp.mpf(repr(float(params[1])))
        r_squared_float = float(fit.rsquared)
        if not math.isfinite(r_squared_float):
            r_squared_float = 0.0
        residuals = [mp.mpf(repr(float(value))) for value in np.asarray(fit.resid, dtype=float)]
        if len(residuals) <= 2:
            fail(
                f"invalid portfolio windows: OLS residual degrees of freedom for {field_name}"
            )
        residual_std = mp.sqrt(
            mp.fsum(residual ** 2 for residual in residuals)
            / mp.mpf(len(residuals) - 2)
        )
        if greater_than(alpha, mp.mpf("0")):
            trend = "positive_alpha"
        elif less_than(alpha, mp.mpf("0")):
            trend = "negative_alpha"
        else:
            trend = "flat_alpha"
        return {
            "alpha": round_float(alpha),
            "beta": round_float(beta),
            "r_squared": round_float(mp.mpf(repr(r_squared_float))),
            "residual_std": round_float(residual_std),
            "trend_label": trend,
        }

    portfolio_metrics = OrderedDict()
    portfolio_raw = {}
    for agent in AGENTS:
        for risk in RISK_PROFILES:
            selected = [
                ticker
                for ticker in tickers
                if decision_table[(ticker, agent, risk)]["decision"] == "BUY"
            ]
            key = f"{agent}|{risk}"
            returns = equal_weight_returns(selected, key)
            cumulative = compound_return(returns)
            volatility = mp.sqrt(mp.mpf(trading_days)) * mp_sample_std(
                returns, f"{key} returns"
            )
            drawdown = max_drawdown(returns)
            sharpe = sharpe_ratio(returns, risk_free_returns, key)
            rolling = rolling_sharpe_values(returns, risk_free_returns, key)
            finite_rolling = [value for value in rolling if value is not None]
            portfolio_metrics[key] = {
                "selected_tickers": selected,
                "selected_count": int(len(selected)),
                "cumulative_return": round_float(cumulative),
                "annualized_volatility": round_float(volatility),
                "max_drawdown": round_float(drawdown),
                "sharpe_ratio": round_float(sharpe),
                "benchmark_excess_return": round_float(cumulative - benchmark_cumulative),
                "rolling_sharpe_min": round_float(
                    min(finite_rolling) if finite_rolling else None
                ),
                "rolling_sharpe_max": round_float(
                    max(finite_rolling) if finite_rolling else None
                ),
                "rolling_sharpe_last": round_float(rolling[-1] if rolling else None),
                "ols_excess_return_model": ols_diagnostics(
                    returns,
                    benchmark_returns,
                    risk_free_returns,
                    key,
                ),
            }
            portfolio_raw[key] = {
                "cumulative_return": cumulative,
                "annualized_volatility": volatility,
                "max_drawdown_magnitude": abs(drawdown),
                "sharpe_ratio": sharpe,
            }

    def portfolio_order_index(name):
        agent, risk = name.split("|", 1)
        return (AGENTS.index(agent), RISK_PROFILES.index(risk))

    def metric_sort(metric_name, higher_is_better):
        names = list(portfolio_metrics.keys())

        def cmp(left, right):
            a = portfolio_raw[left][metric_name]
            b = portfolio_raw[right][metric_name]
            if a is None and b is None:
                return (
                    -1
                    if portfolio_order_index(left) < portfolio_order_index(right)
                    else (
                        1
                        if portfolio_order_index(left) > portfolio_order_index(right)
                        else 0
                    )
                )
            if a is None:
                return 1
            if b is None:
                return -1
            if abs(a - b) <= tol:
                return (
                    -1
                    if portfolio_order_index(left) < portfolio_order_index(right)
                    else (
                        1
                        if portfolio_order_index(left) > portfolio_order_index(right)
                        else 0
                    )
                )
            if higher_is_better:
                return -1 if a > b else 1
            return -1 if a < b else 1

        return sorted(names, key=cmp_to_key(cmp))

    rankings = {
        "by_cumulative_return_desc": metric_sort("cumulative_return", True),
        "by_sharpe_desc": metric_sort("sharpe_ratio", True),
        "by_drawdown_asc": metric_sort("max_drawdown_magnitude", False),
        "by_volatility_asc": metric_sort("annualized_volatility", False),
    }

    def selected_set(agent, risk):
        return [
            ticker
            for ticker in tickers
            if decision_table[(ticker, agent, risk)]["decision"] == "BUY"
        ]

    def jaccard(left, right):
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        if not union:
            return mp.mpf("1")
        return mp.mpf(len(left_set & right_set)) / mp.mpf(len(union))

    def feature_means(selected, field_name):
        if not selected:
            fail(f"empty portfolios where a metric is required: {field_name}")
        return {
            col: mp_mean(
                [feature_map[ticker][col] for ticker in selected],
                f"{field_name}.{col}",
            )
            for col in feature_columns
        }

    agent_risk_profiles = OrderedDict()
    risk_feature_raw_by_agent = OrderedDict()
    total_monotonicity_violation_count = 0
    projection_success_count = 0
    projection_non_skipped_count = 0

    for agent in AGENTS:
        selected_sets = OrderedDict(
            (risk, selected_set(agent, risk)) for risk in RISK_PROFILES
        )
        risk_feature_raw = OrderedDict(
            (risk, feature_means(selected_sets[risk], f"{agent}|{risk}"))
            for risk in RISK_PROFILES
        )
        risk_feature_raw_by_agent[agent] = risk_feature_raw
        risk_feature_means = OrderedDict()
        for risk in RISK_PROFILES:
            risk_feature_means[risk] = {
                "january_return": round_float(risk_feature_raw[risk]["january_return"]),
                "january_annualized_volatility": round_float(
                    risk_feature_raw[risk]["january_annualized_volatility"]
                ),
                "january_max_drawdown": round_float(
                    risk_feature_raw[risk]["january_max_drawdown"]
                ),
                "january_beta_to_equal_weight_universe": round_float(
                    risk_feature_raw[risk]["january_beta_to_equal_weight_universe"]
                ),
                "january_average_volume": round_float(
                    risk_feature_raw[risk]["january_average_volume"]
                ),
            }
        neutral = risk_feature_raw["risk_neutral"]
        averse = risk_feature_raw["risk_averse"]
        violations = []
        if greater_than(
            averse["january_annualized_volatility"],
            neutral["january_annualized_volatility"],
        ):
            violations.append("higher_average_volatility")
        if greater_than(
            averse["january_beta_to_equal_weight_universe"],
            neutral["january_beta_to_equal_weight_universe"],
        ):
            violations.append("higher_average_beta")
        if less_than(
            averse["january_max_drawdown"],
            neutral["january_max_drawdown"],
        ):
            violations.append("worse_average_drawdown")
        total_monotonicity_violation_count += len(violations)

        jaccard_values = OrderedDict()
        jaccard_values["risk_neutral|risk_averse"] = round_float(
            jaccard(selected_sets["risk_neutral"], selected_sets["risk_averse"])
        )
        jaccard_values["risk_neutral|risk_seeking"] = round_float(
            jaccard(selected_sets["risk_neutral"], selected_sets["risk_seeking"])
        )
        jaccard_values["risk_averse|risk_seeking"] = round_float(
            jaccard(selected_sets["risk_averse"], selected_sets["risk_seeking"])
        )
        risk_seeking_similarity = jaccard(
            selected_sets["risk_neutral"], selected_sets["risk_seeking"]
        )

        neutral_selected = selected_sets["risk_neutral"]
        if not neutral_selected:
            projection = {
                "solver": "skipped_empty_neutral_set",
                "status": "skipped_empty_neutral_set",
                "objective_value": None,
                "projected_average_volatility": None,
                "max_absolute_decision_shift": None,
            }
        else:
            projection_non_skipped_count += 1
            vols = np.array(
                [
                    float(feature_map[ticker]["january_annualized_volatility"])
                    for ticker in tickers
                ],
                dtype=float,
            )
            b_values = np.array(
                [
                    1.0 if ticker in selected_sets["risk_averse"] else 0.0
                    for ticker in tickers
                ],
                dtype=float,
            )
            neutral_average_volatility = float(
                mp_mean(
                    [
                        feature_map[ticker]["january_annualized_volatility"]
                        for ticker in neutral_selected
                    ],
                    f"{agent}.neutral_volatility",
                )
            )
            x = cp.Variable(len(tickers))
            problem = cp.Problem(
                cp.Minimize(cp.sum_squares(x - b_values)),
                [
                    x >= 0,
                    x <= 1,
                    cp.sum(x) >= 1,
                    (vols - neutral_average_volatility) @ x <= 0,
                ],
            )
            solver_used = None
            status = None
            for solver_name in ["OSQP", "SCS"]:
                try:
                    problem.solve(solver=getattr(cp, solver_name), verbose=False)
                    solver_used = solver_name
                    status = str(problem.status)
                except Exception:
                    solver_used = solver_name
                    status = "solver_error"
                if status in {"optimal", "optimal_inaccurate"}:
                    break
            if status in {"optimal", "optimal_inaccurate"}:
                projection_success_count += 1
            if x.value is None or status not in {"optimal", "optimal_inaccurate"}:
                projection = {
                    "solver": solver_used,
                    "status": status,
                    "objective_value": None,
                    "projected_average_volatility": None,
                    "max_absolute_decision_shift": None,
                }
            else:
                x_values = [
                    mp.mpf(repr(float(value))) for value in np.asarray(x.value).ravel()
                ]
                sum_x = mp.fsum(x_values)
                if sum_x <= 0:
                    fail("invalid convex projection: non-positive projected inclusion sum")
                objective_value = mp.fsum(
                    (x_values[i] - mp.mpf(repr(float(b_values[i])))) ** 2
                    for i in range(len(tickers))
                )
                projected_average_volatility = (
                    mp.fsum(
                        x_values[i]
                        * feature_map[tickers[i]]["january_annualized_volatility"]
                        for i in range(len(tickers))
                    )
                    / sum_x
                )
                max_shift = max(
                    abs(x_values[i] - mp.mpf(repr(float(b_values[i]))))
                    for i in range(len(tickers))
                )
                projection = {
                    "solver": solver_used,
                    "status": status,
                    "objective_value": round_float(objective_value),
                    "projected_average_volatility": round_float(
                        projected_average_volatility
                    ),
                    "max_absolute_decision_shift": round_float(max_shift),
                }

        agent_risk_profiles[agent] = {
            "selected_sets": selected_sets,
            "jaccard": jaccard_values,
            "risk_feature_means": risk_feature_means,
            "monotonicity_violations": violations,
            "risk_seeking_similarity_to_neutral": round_float(risk_seeking_similarity),
            "cvxpy_projection": projection,
        }

    total_claims = sum(claim_status_counts.values())
    evidence_score = (
        mp.mpf("1")
        if total_claims == 0
        else mp.mpf(
            claim_status_counts["supported"] + claim_status_counts["missing_evidence"]
        )
        / mp.mpf(total_claims)
    )
    debate_score = global_debate_score
    finite_portfolio_count = 0
    for metrics in portfolio_metrics.values():
        checked = [
            metrics["cumulative_return"],
            metrics["annualized_volatility"],
            metrics["max_drawdown"],
            metrics["sharpe_ratio"],
        ]
        if all(value is not None and math.isfinite(float(value)) for value in checked):
            finite_portfolio_count += 1
    portfolio_score = mp.mpf(finite_portfolio_count) / mp.mpf(
        len(AGENTS) * len(RISK_PROFILES)
    )
    monotonicity_checks = len(AGENTS) * 3
    risk_profile_score = mp.mpf("1") - mp.mpf(
        total_monotonicity_violation_count
    ) / mp.mpf(monotonicity_checks)
    if risk_profile_score < 0:
        risk_profile_score = mp.mpf("0")
    optimization_score = (
        mp.mpf("1")
        if projection_non_skipped_count == 0
        else mp.mpf(projection_success_count) / mp.mpf(projection_non_skipped_count)
    )
    global_consistency_score = (
        evidence_score
        + debate_score
        + portfolio_score
        + risk_profile_score
        + optimization_score
    ) / mp.mpf("5")

    all_claims_supported_or_missing = (
        claim_status_counts["role_access_violation"] == 0
        and claim_status_counts["decision_mismatch"] == 0
        and claim_status_counts["unsupported"] == 0
    )
    all_debates_valid = all(row["termination_valid"] for row in debate_level_results.values())
    all_consensus_reconciled = all(
        row["consensus_matches_decision_table"] for row in debate_level_results.values()
    )
    multi_agent_risk_neutral_outperforms = greater_than(
        portfolio_raw["multi_agent|risk_neutral"]["cumulative_return"],
        benchmark_cumulative,
    )
    risk_averse_lower_volatility = True
    for agent in AGENTS:
        if greater_than(
            risk_feature_raw_by_agent[agent]["risk_averse"][
                "january_annualized_volatility"
            ],
            risk_feature_raw_by_agent[agent]["risk_neutral"][
                "january_annualized_volatility"
            ],
        ):
            risk_averse_lower_volatility = False
            break
    cvxpy_projection_all_feasible = projection_non_skipped_count == projection_success_count

    component_scores = [
        evidence_score,
        debate_score,
        portfolio_score,
        risk_profile_score,
        optimization_score,
    ]
    if all(abs(score - 1) <= tol for score in component_scores):
        audit_signature = "fully_consistent"
        audit_signature_code = 3
    elif (
        abs(portfolio_score - 1) <= tol
        and abs(debate_score - 1) <= tol
        and (evidence_score < 1 - tol or risk_profile_score < 1 - tol)
    ):
        audit_signature = "portfolio_consistent_with_evidence_warnings"
        audit_signature_code = 2
    elif total_monotonicity_violation_count > 0 and all_consensus_reconciled:
        audit_signature = "risk_profile_inconsistent"
        audit_signature_code = 1
    else:
        audit_signature = "globally_inconsistent"
        audit_signature_code = 0

    result = OrderedDict()
    result["dataset_profile"] = {
        "ticker_count": int(len(tickers)),
        "agent_count": int(len(AGENTS)),
        "risk_profile_count": int(len(RISK_PROFILES)),
        "decision_count": int(len(decisions_df)),
        "report_count": int(len(reports_map)),
        "claim_count": int(total_claims),
        "debate_count": int(debate_count),
        "price_observation_count": int(len(price_records)),
        "portfolio_trading_day_count": int(len(return_dates)),
    }
    result["evidence_audit"] = {
        "claim_status_counts": claim_status_counts,
        "agent_claim_scores": agent_claim_scores,
        "claim_level_results": claim_level_results,
    }
    result["debate_audit"] = {
        "debate_level_results": debate_level_results,
        "global_debate_score": round_float(global_debate_score),
        "invalid_debate_count": int(invalid_debate_count),
        "unresolved_dissent_count": int(unresolved_dissent_count),
    }
    result["portfolio_audit"] = {
        "portfolio_metrics": portfolio_metrics,
        "benchmark_metrics": {
            "cumulative_return": round_float(benchmark_cumulative),
            "annualized_volatility": round_float(benchmark_volatility),
            "max_drawdown": round_float(benchmark_drawdown),
            "sharpe_ratio": round_float(benchmark_sharpe),
        },
        "rankings": rankings,
    }
    result["risk_profile_audit"] = {
        "agent_risk_profiles": agent_risk_profiles,
        "total_monotonicity_violation_count": int(total_monotonicity_violation_count),
    }
    result["summary"] = {
        "all_claims_supported_or_missing_by_design": bool(
            all_claims_supported_or_missing
        ),
        "all_debates_valid": bool(all_debates_valid),
        "all_consensus_decisions_reconciled": bool(all_consensus_reconciled),
        "multi_agent_risk_neutral_outperforms_benchmark": bool(
            multi_agent_risk_neutral_outperforms
        ),
        "risk_averse_has_lower_average_volatility_than_risk_neutral": bool(
            risk_averse_lower_volatility
        ),
        "cvxpy_projection_all_feasible": bool(cvxpy_projection_all_feasible),
        "global_consistency_score": round_float(global_consistency_score),
        "audit_signature": audit_signature,
        "audit_signature_code": int(audit_signature_code),
    }

    try:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except OSError as exc:
        fail(f"write failure: {exc}")
    return result
