import os
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt

def load_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_DATABASE", "esg_risk_db"),
        "user": os.getenv("DB_USER", "user"),
        "password": os.getenv("DB_PASSWORD", "password"),
        "port": os.getenv("DB_PORT", "5432"),
        "sslmode": "require"
    }

# Connect to Azure PostgreSQL
conn = psycopg2.connect(**load_config())
cursor = conn.cursor()

# Create tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS esg_data (
        company VARCHAR,
        sector VARCHAR,
        carbon_emissions FLOAT,
        diversity_ratio FLOAT,
        governance_score FLOAT,
        risk_score FLOAT,
        date DATE
    );
    CREATE TABLE IF NOT EXISTS financial_data (
        company VARCHAR,
        date DATE,
        close_price FLOAT,
        volatility FLOAT,
        debt_to_equity FLOAT
    );
    CREATE TABLE IF NOT EXISTS news_social_data (
        company VARCHAR,
        sector VARCHAR,
        text TEXT,
        sentiment FLOAT,
        date TIMESTAMP,
        source VARCHAR,
        esg_category VARCHAR,
        keywords TEXT
    );
    CREATE TABLE IF NOT EXISTS macro_data (
        date DATE,
        region VARCHAR,
        carbon_tax_rate FLOAT,
        climate_risk_index FLOAT,
        interest_rate FLOAT
    );
    CREATE TABLE IF NOT EXISTS portfolio_data (
        company VARCHAR,
        sector VARCHAR,
        portfolio_weight FLOAT,
        investment_value FLOAT
    );
""")
conn.commit()

# Load data
for table, file in [
    ("esg_data", "data/esg_data.csv"),
    ("financial_data", "data/financial_data.csv"),
    ("news_social_data", "data/news_social_data.csv"),
    ("macro_data", "data/macro_data.csv"),
    ("portfolio_data", "data/portfolio_data.csv")
]:
    df = pd.read_csv(file)
    df.to_sql(table, conn, if_exists="replace", index=False)

# EDA
esg_df = pd.read_csv("data/esg_data.csv")
financial_df = pd.read_csv("data/financial_data.csv")
news_social_df = pd.read_csv("data/news_social_data.csv")
macro_df = pd.read_csv("data/macro_data.csv")
portfolio_df = pd.read_csv("data/portfolio_data.csv")
merged_data = esg_df.merge(financial_df, on=["company", "date"], how="left").merge(
    news_social_df.groupby(["company", "date"])["sentiment"].mean().reset_index(), on=["company", "date"], how="left"
).merge(portfolio_df, on=["company", "sector"]).merge(macro_df, on="date")
merged_data.fillna(method="ffill", inplace=True)
merged_data.to_csv("data/merged_data.csv", index=False)

# Visualizations
sns.heatmap(merged_data.corr(), annot=True, cmap="coolwarm")
plt.savefig("static/correlation_heatmap.png")
plt.close()
sns.histplot(merged_data["carbon_emissions"], bins=30)
plt.savefig("static/emissions_distribution.png")
plt.close()

cursor.close()
conn.close()