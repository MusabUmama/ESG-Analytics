# src/generate_data.py
import json
import os
from together import Together
import pandas as pd
import numpy as np
import spacy
import logging
from dotenv import load_dotenv
import time
import re

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
if not os.getenv("TOGETHER_API_KEY"):
    logger.error("TOGETHER_API_KEY not set")
    raise ValueError("TOGETHER_API_KEY environment variable is required")
np.random.seed(42)
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Define companies
companies = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Tech"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Tech"},
    {"ticker": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Finance"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Tech"},
    {"ticker": "WMT", "name": "Walmart Inc.", "sector": "Retail"},
    {"ticker": "CVX", "name": "Chevron Corporation", "sector": "Energy"},
    {"ticker": "BAC", "name": "Bank of America Corporation", "sector": "Finance"},
    {"ticker": "NKE", "name": "Nike, Inc.", "sector": "Consumer Goods"},
    {"ticker": "TSLA", "name": "Tesla, Inc.", "sector": "Automotive"}
]
company_df = pd.DataFrame(companies)
os.makedirs("data", exist_ok=True)
company_df.to_csv("data/companies.csv", index=False)
logger.info("Generated data/companies.csv")

# Helper function to extract JSON from response
def extract_json(text):
    try:
        # Extract content between first { and last }
        match = re.search(r'\{[\s\S]*?\}', text)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        logger.warning(f"No JSON found in response: {text}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}, text: {text}")
        return None

# ESG Data
dates = pd.date_range(start="2020-01-01", end="2025-08-06", freq="QE")
esg_data = []
for company in companies:
    sector = company["sector"]
    prompt = f"""
[INST] Generate realistic ESG metrics for {company['name']} ({sector} sector) for a single quarter.
Return *only* a JSON object with the following fields:
- carbon_emissions: float (50000–5000000)
- diversity_ratio: float (0–1)
- governance_score: float (0–100)
Example: {{"carbon_emissions": 100000.0, "diversity_ratio": 0.4, "governance_score": 85.0}}
Output must be valid JSON, enclosed in {{}}, with no additional text, explanations, or code fences (```). [/INST]
"""
    for date in dates:
        try:
            response = client.completions.create(model=model, prompt=prompt, max_tokens=100, temperature=0.0)
            time.sleep(1)
            response_text = response.choices[0].text.strip()
            logger.debug(f"ESG response for {company['name']} on {date}: {response_text}")
            metrics = extract_json(response_text)
            if not metrics or not all(key in metrics for key in ["carbon_emissions", "diversity_ratio", "governance_score"]):
                raise ValueError("Incomplete or invalid JSON response")
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.warning(f"Failed to parse ESG metrics for {company['name']} on {date}: {e}")
            metrics = {}
        esg_data.append({
            "company": company["ticker"],
            "sector": sector,
            "carbon_emissions": metrics.get("carbon_emissions", np.random.uniform(50000, 5000000)),
            "diversity_ratio": metrics.get("diversity_ratio", np.random.uniform(0.1, 0.5)),
            "governance_score": metrics.get("governance_score", np.random.uniform(60, 95)),
            "risk_score": np.random.uniform(20, 80),
            "date": date
        })
esg_df = pd.DataFrame(esg_data)
esg_df.to_csv("data/esg_data.csv", index=False)
logger.info("Generated data/esg_data.csv")

# Financial Data
financial_data = []
for company in companies:
    prompt = f"""
[INST] Generate realistic financial metrics for {company['name']} ({company['sector']}) for a single quarter.
Return *only* a JSON object with the following fields:
- stock_price: float (USD)
- volatility: float (0–0.1)
- debt_to_equity: float (0.2–2.0)
Example: {{"stock_price": 150.0, "volatility": 0.02, "debt_to_equity": 0.5}}
Output must be valid JSON, enclosed in {{}}, with no additional text, explanations, or code fences (```). [/INST]
"""
    prices = [100]
    for _ in range(len(dates) - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    for date, price in zip(dates, prices):
        try:
            response = client.completions.create(model=model, prompt=prompt, max_tokens=100, temperature=0.0)
            time.sleep(1)
            response_text = response.choices[0].text.strip()
            logger.debug(f"Financial response for {company['name']} on {date}: {response_text}")
            metrics = extract_json(response_text)
            if not metrics or not all(key in metrics for key in ["stock_price", "volatility", "debt_to_equity"]):
                raise ValueError("Incomplete or invalid JSON response")
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.warning(f"Failed to parse financial metrics for {company['name']} on {date}: {e}")
            metrics = {}
        financial_data.append({
            "company": company["ticker"],
            "date": date,
            "close_price": metrics.get("stock_price", price),
            "volatility": metrics.get("volatility", np.std(np.diff(prices[-4:])) / np.mean(prices[-4:]) if len(prices) >= 4 else 0.02),
            "debt_to_equity": metrics.get("debt_to_equity", np.random.uniform(0.2, 2.0))
        })
financial_df = pd.DataFrame(financial_data)
financial_df.to_csv("data/financial_data.csv", index=False)
logger.info("Generated data/financial_data.csv")

# News/Social Media Data
prompts = {
    "Environmental": f"[INST] Generate a realistic news headline about {{company}}’s environmental impact (e.g., emissions, sustainability). Return a single string, no JSON, no code fences (```). [/INST]",
    "Social": f"[INST] Generate a realistic X post about {{company}}’s labor practices or diversity. Return a single string, no JSON, no code fences (```). [/INST]",
    "Governance": f"[INST] Generate a realistic news headline about {{company}}’s governance practices (e.g., board transparency). Return a single string, no JSON, no code fences (```). [/INST]"
}
news_social_data = []
dates = pd.date_range("2025-08-01", "2025-08-06", freq="H")[:50]
for company in companies:
    for date in dates:
        for category, prompt in prompts.items():
            try:
                response = client.completions.create(model=model, prompt=prompt.format(company=company["name"]), max_tokens=50, temperature=0.0)
                time.sleep(1)
                text = response.choices[0].text.strip()
                sentiment_prompt = f"[INST] Analyze sentiment of '{text}' (-1 to 1). Return a single float value, no JSON, no code fences (```). [/INST]"
                sentiment_response = client.completions.create(model=model, prompt=sentiment_prompt, max_tokens=10, temperature=0.0)
                time.sleep(1)
                sentiment_text = sentiment_response.choices[0].text.strip()
                sentiment = float(sentiment_text) if sentiment_text.replace(".", "").replace("-", "").isdigit() else np.random.uniform(-1, 1)
            except Exception as e:
                logger.warning(f"Failed to generate news/social data for {company['name']} on {date}: {e}")
                text = "Sample text"
                sentiment = np.random.uniform(-1, 1)
            news_social_data.append({
                "company": company["ticker"],
                "sector": company["sector"],
                "text": text,
                "sentiment": sentiment,
                "date": date,
                "source": "X" if "X post" in prompt else "News",
                "esg_category": category,
                "keywords": [t.text for t in nlp(text.lower()) if t.text in ["emissions", "labor", "diversity", "governance"]]
            })
news_social_df = pd.DataFrame(news_social_data)
news_social_df.to_csv("data/news_social_data.csv", index=False)
logger.info("Generated data/news_social_data.csv")

# Macro Indicators
macro_data = []
dates = pd.date_range(start="2020-01-01", end="2025-08-06", freq="QE")
for date in dates:
    prompt = f"""
[INST] Generate realistic US macro indicators for a single quarter.
Return *only* a JSON object with the following fields:
- carbon_tax_rate: float (0.02–0.08)
- climate_risk_index: float (0.1–0.5)
- interest_rate: float (0.01–0.05)
Example: {{"carbon_tax_rate": 0.05, "climate_risk_index": 0.3, "interest_rate": 0.03}}
Output must be valid JSON, enclosed in {{}}, with no additional text, explanations, or code fences (```). [/INST]
"""
    try:
        response = client.completions.create(model=model, prompt=prompt, max_tokens=100, temperature=0.0)
        time.sleep(1)
        response_text = response.choices[0].text.strip()
        logger.debug(f"Macro response for {date}: {response_text}")
        metrics = extract_json(response_text)
        if not metrics or not all(key in metrics for key in ["carbon_tax_rate", "climate_risk_index", "interest_rate"]):
            raise ValueError("Incomplete or invalid JSON response")
    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.warning(f"Failed to parse macro indicators for {date}: {e}")
        metrics = {}
    macro_data.append({
        "date": date,
        "region": "US",
        "carbon_tax_rate": metrics.get("carbon_tax_rate", np.random.uniform(0.02, 0.08)),
        "climate_risk_index": metrics.get("climate_risk_index", np.random.uniform(0.1, 0.5)),
        "interest_rate": metrics.get("interest_rate", np.random.uniform(0.01, 0.05))
    })
macro_df = pd.DataFrame(macro_data)
macro_df.to_csv("data/macro_data.csv", index=False)
logger.info("Generated data/macro_data.csv")

# Portfolio Data
weights = np.random.dirichlet(np.ones(len(companies)), size=1)[0]
portfolio_data = [
    {"company": c["ticker"], "sector": c["sector"], "portfolio_weight": w, "investment_value": w * 10000000}
    for c, w in zip(companies, weights)
]
portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df.to_csv("data/portfolio_data.csv", index=False)
logger.info("Generated data/portfolio_data.csv")