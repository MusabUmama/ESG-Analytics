# src/preprocess_data.py
import os
import psycopg2
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import ast

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

# Log column names for debugging
logger.debug(f"company_df columns: {list(company_df.columns)}")
logger.debug(f"esg_df columns: {list(esg_df.columns)}")
logger.debug(f"financial_df columns: {list(financial_df.columns)}")
logger.debug(f"portfolio_df columns: {list(portfolio_df.columns)}")
logger.debug(f"macro_df columns: {list(macro_df.columns)}")
logger.debug(f"news_social_df columns: {list(news_social_df.columns)}")

# Rename 'ticker' to 'company' in company_df for consistency
if "ticker" in company_df.columns:
    company_df = company_df.rename(columns={"ticker": "company"})
    logger.info("Renamed 'ticker' to 'company' in company_df")

# Convert stringified 'keywords' to lists
try:
    news_social_df["keywords"] = news_social_df["keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    logger.info("Converted 'keywords' column to lists")
except (ValueError, SyntaxError) as e:
    logger.warning(f"Failed to convert some 'keywords' values: {e}")
    news_social_df["keywords"] = news_social_df["keywords"].apply(lambda x: [] if pd.isna(x) or x == "" else x)

# Preprocess data
try:
    # Convert dates to datetime
    esg_df["date"] = pd.to_datetime(esg_df["date"])
    financial_df["date"] = pd.to_datetime(financial_df["date"])
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    news_social_df["date"] = pd.to_datetime(news_social_df["date"])

    # Merge data
    merged_df = esg_df.merge(financial_df, on=["company", "date"], how="left")
    merged_df = merged_df.merge(portfolio_df, on=["company", "sector"], how="left")
    merged_df = merged_df.merge(macro_df, on=["date"], how="left")
    merged_df = merged_df.merge(company_df, on=["company", "sector"], how="left")

    # Handle missing values
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

    # Aggregate news/social data
    news_social_agg = news_social_df.groupby(["company", "date"]).agg({
        "sentiment": "mean",
        "keywords": lambda x: list(set([item for sublist in x if isinstance(sublist, list) for item in sublist]))
    }).reset_index()
    merged_df = merged_df.merge(news_social_agg, on=["company", "date"], how="left")
    merged_df["sentiment"] = merged_df["sentiment"].fillna(0)

    # Save merged data
    os.makedirs("data", exist_ok=True)
    merged_df.to_csv("data/merged_data.csv", index=False)
    logger.info("Generated data/merged_data.csv")
except Exception as e:
    logger.error(f"Error during preprocessing: {e}")
    raise

# Connect to Supabase
try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    logger.info("Connected to Supabase")

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS esg_data (
            id SERIAL PRIMARY KEY,
            company VARCHAR(10),
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
            sentiment FLOAT
        );
    """)

    # Insert data
    for _, row in merged_df.iterrows():
        cursor.execute("""
            INSERT INTO esg_data (
                company, sector, date, carbon_emissions, diversity_ratio, governance_score,
                risk_score, close_price, volatility, debt_to_equity, portfolio_weight,
                investment_value, carbon_tax_rate, climate_risk_index, interest_rate, sentiment
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            row["company"], row["sector"], row["date"], row["carbon_emissions"],
            row["diversity_ratio"], row["governance_score"], row["risk_score"],
            row["close_price"], row["volatility"], row["debt_to_equity"],
            row["portfolio_weight"], row["investment_value"], row["carbon_tax_rate"],
            row["climate_risk_index"], row["interest_rate"], row["sentiment"]
        ))

    conn.commit()
    logger.info("Inserted data into Supabase")
except Exception as e:
    logger.error(f"Failed to connect or insert into Supabase: {e}")
    raise
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    logger.info("Closed Supabase connection")