# src/preprocess_data.py
import os
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import ast
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL not set in .env")
    raise ValueError("DATABASE_URL environment variable is required")

# Load data
try:
    company_df = pd.read_csv("src/data/companies.csv")
    esg_df = pd.read_csv("src/data/esg_data.csv")
    financial_df = pd.read_csv("src/data/financial_data.csv")
    news_social_df = pd.read_csv("src/data/news_social_data.csv")
    macro_df = pd.read_csv("src/data/macro_data.csv")
    portfolio_df = pd.read_csv("src/data/portfolio_data.csv")
    logger.info("Loaded all CSV files")
except FileNotFoundError as e:
    logger.error(f"CSV file not found: {e}")
    raise

if "ticker" in company_df.columns:
    company_df = company_df.rename(columns={"ticker": "company"})
    logger.info("Renamed 'ticker' to 'company' in company_df")

# Convert keywords from string to list
try:
    news_social_df["keywords"] = news_social_df["keywords"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    logger.info("Converted 'keywords' column to lists")
except (ValueError, SyntaxError) as e:
    logger.warning(f"Failed to convert some 'keywords' values: {e}")
    news_social_df["keywords"] = news_social_df["keywords"].apply(
        lambda x: [] if pd.isna(x) or x == "" else x
    )

# Preprocess data
try:
    # Convert dates
    esg_df["date"] = pd.to_datetime(esg_df["date"])
    financial_df["date"] = pd.to_datetime(financial_df["date"])
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    news_social_df["date"] = pd.to_datetime(news_social_df["date"])

    # Merge datasets
    merged_df = esg_df.merge(financial_df, on=["company", "date"], how="left")
    merged_df = merged_df.merge(portfolio_df, on=["company", "sector"], how="left")
    merged_df = merged_df.merge(macro_df, on=["date"], how="left")
    merged_df = merged_df.merge(company_df, on=["company", "sector"], how="left")

    # Ensure region exists
    if "region" not in merged_df.columns:
        merged_df["region"] = "Unknown"

    # Fill missing values
    merged_df.fillna({
        "carbon_emissions": merged_df["carbon_emissions"].mean(),
        "diversity_ratio": merged_df["diversity_ratio"].mean(),
        "governance_score": merged_df["governance_score"].mean(),
        "risk_score": merged_df["risk_score"].mean(),
        "close_price": merged_df["close_price"].mean(),
        "volatility": merged_df["volatility"].mean(),
        "debt_to_equity": merged_df["debt_to_equity"].mean(),
        "portfolio_weight": merged_df["portfolio_weight"].mean(),
        "investment_value": merged_df["investment_value"].mean(),
        "carbon_tax_rate": merged_df["carbon_tax_rate"].mean(),
        "climate_risk_index": merged_df["climate_risk_index"].mean(),
        "interest_rate": merged_df["interest_rate"].mean()
    }, inplace=True)

    # Aggregate news/social
    news_social_agg = news_social_df.groupby(["company", "date"]).agg({
        "sentiment": "mean",
        "keywords": lambda x: list(set([item for sublist in x if isinstance(sublist, list) for item in sublist]))
    }).reset_index()

    merged_df = merged_df.merge(news_social_agg, on=["company", "date"], how="left")
    merged_df["sentiment"] = merged_df["sentiment"].fillna(0)
    merged_df["keywords"] = merged_df["keywords"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    merged_df["carbon_cost_est"] = merged_df["carbon_tax_rate"] * merged_df["carbon_emissions"]
    merged_df["sentiment_missing"] = merged_df["sentiment"].isna().astype(int)
    merged_df["log_carbon_emissions"] = np.log1p(merged_df["carbon_emissions"])
    merged_df["log_close_price"] = np.log1p(merged_df["close_price"])
    merged_df["log_investment_value"] = np.log1p(merged_df["investment_value"])
    merged_df["vol_x_dte"] = merged_df["volatility"] * merged_df["debt_to_equity"]
    merged_df["keywords_len"] = merged_df["keywords"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)

    os.makedirs("data", exist_ok=True)
    merged_df.to_csv("data/merged_data.csv", index=False)
    logger.info("Generated data/merged_data.csv with engineered features")
except Exception as e:
    logger.error(f"Error during preprocessing: {e}")
    raise

# Push to Supabase
try:
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS esg_data (
                id SERIAL PRIMARY KEY,
                company VARCHAR(20),
                sector VARCHAR(50),
                date TIMESTAMP,
                carbon_emissions FLOAT,
                diversity_ratio FLOAT,
                governance_score FLOAT,
                risk_score FLOAT,
                close_price FLOAT,
                volatility FLOAT,
                debt_to_equity FLOAT,
                portfolio_weight FLOAT,
                investment_value FLOAT,
                carbon_tax_rate FLOAT,
                climate_risk_index FLOAT,
                interest_rate FLOAT,
                sentiment FLOAT,
                keywords TEXT,
                region TEXT,
                name TEXT,
                carbon_cost_est FLOAT,
                sentiment_missing INT,
                log_carbon_emissions FLOAT,
                log_close_price FLOAT,
                log_investment_value FLOAT,
                vol_x_dte FLOAT,
                keywords_len INT
            );
        """))

        conn.execute(text("TRUNCATE esg_data;"))

    merged_df.to_sql(
        "esg_data",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500
    )
    logger.info("Inserted data with engineered features into Supabase successfully")
except Exception as e:
    logger.error(f"Failed to insert into Supabase: {e}")
    raise
