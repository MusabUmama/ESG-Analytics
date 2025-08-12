import json
import os
from together import Together
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
np.random.seed(42)
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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
company_df.to_csv("data/companies.csv", index=False)

# ESG Data
dates = pd.date_range(start="2020-01-01", end="2025-08-06", freq="Q")
esg_data = []
for company in companies:
    sector = company["sector"]
    prompt = f"Generate realistic ESG metrics for {company['name']} ({sector} sector) including carbon emissions (tons), diversity ratio (0–1), and governance score (0–100) in JSON format."
    for date in dates:
        response = client.completions.create(model=model, prompt=prompt, max_tokens=100)
        try:
            metrics = json.loads(response.choices[0].text.strip())
        except json.JSONDecodeError:
            metrics = {}  # Fallback to defaults if JSON parsing fails
        esg_data.append({
            "company": company["ticker"],
            "sector": sector,
            "carbon_emissions": metrics.get("carbon_emissions", np.random.uniform(50000, 5000000)),
            "diversity_ratio": metrics.get("diversity_ratio", np.random.uniform(0.1, 0.5)),
            "governance_score": metrics.get("governance_score", np.random.uniform(60, 95)),
            "risk_score": np.random.uniform(20, 80),  # Placeholder
            "date": date
        })
esg_df = pd.DataFrame(esg_data)
esg_df.to_csv("data/esg_data.csv", index=False)

# Financial Data
financial_data = []
for company in companies:
    prompt = f"Generate realistic financial metrics for {company['name']} ({company['sector']}) including stock price, volatility, and debt-to-equity ratio in JSON format."
    prices = [100]
    for _ in range(len(dates) - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    for date, price in zip(dates, prices):
        response = client.completions.create(model=model, prompt=prompt, max_tokens=100)
        try:
            metrics = json.loads(response.choices[0].text.strip())
        except json.JSONDecodeError:
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

# News/Social Media Data
prompts = {
    "Environmental": "Generate a realistic news headline about {company}’s environmental impact (e.g., emissions, sustainability).",
    "Social": "Generate a realistic X post about {company}’s labor practices or diversity.",
    "Governance": "Generate a realistic news headline about {company}’s governance practices (e.g., board transparency)."
}
news_social_data = []
dates = pd.date_range("2025-08-01", "2025-08-06", freq="H")[:50]
for company in companies:
    for date in dates:
        for category, prompt in prompts.items():
            response = client.completions.create(model=model, prompt=prompt.format(company=company["name"]), max_tokens=50)
            text = response.choices[0].text.strip()
            sentiment_response = client.completions.create(model=model, prompt=f"Analyze sentiment of '{text}' (-1 to 1).", max_tokens=10)
            sentiment = float(sentiment_response.choices[0].text.strip()) if sentiment_response.choices[0].text.strip() else np.random.uniform(-1, 1)
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

# Macro Indicators
macro_data = []
dates = pd.date_range(start="2020-01-01", end="2025-08-06", freq="Q")
for date in dates:
    prompt = "Generate realistic US macro indicators including carbon tax rate, climate risk index, and interest rate in JSON format."
    response = client.completions.create(model=model, prompt=prompt, max_tokens=100)
    try:
        metrics = json.loads(response.choices[0].text.strip())
    except json.JSONDecodeError:
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

# Portfolio Data
weights = np.random.dirichlet(np.ones(len(companies)), size=1)[0]
portfolio_data = [
    {"company": c["ticker"], "sector": c["sector"], "portfolio_weight": w, "investment_value": w * 10000000}
    for c, w in zip(companies, weights)
]
portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df.to_csv("data/portfolio_data.csv", index=False)