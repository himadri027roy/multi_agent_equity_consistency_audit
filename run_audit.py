from src.multi_agent_equity_consistency_audit import multi_agent_equity_consistency_audit


if __name__ == "__main__":
    result = multi_agent_equity_consistency_audit(
        universe_path="data/sample/universe.csv",
        config_path="data/sample/config.json",
        feature_schema_path="data/sample/feature_schema.json",
        prices_path="data/sample/prices.csv",
        risk_free_path="data/sample/risk_free.csv",
        stock_features_path="data/sample/stock_features.csv",
        price_evidence_path="data/sample/price_evidence.jsonl",
        filings_metadata_path="data/sample/filings_metadata.csv",
        filings_chunks_path="data/sample/filings_chunks.jsonl",
        news_articles_path="data/sample/news_articles.jsonl",
        agent_reports_path="data/sample/agent_reports.jsonl",
        claim_evidence_links_path="data/sample/claim_evidence_links.csv",
        debate_logs_path="data/sample/debate_logs.jsonl",
        agent_decisions_path="data/sample/agent_decisions.csv",
        output_path="results/output.json",
    )

    print("Audit completed.")
    print("Global consistency score:", result["summary"]["global_consistency_score"])
    print("Audit signature:", result["summary"]["audit_signature"])
