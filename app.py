# app.py
from flask import Flask, render_template, request, send_file
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.io as pio
import joblib
import os
from kafka import KafkaConsumer
import json
from configparser import ConfigParser
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_config(filename="config/database.ini", section="postgresql"):
    parser = ConfigParser()
    parser.read(filename)
    config = {}
    if parser.has_section(section):
        for item in parser.items(section):
            config[item[0]] = os.getenv(item[1].strip("%"), item[1].strip("%"))
    else:
        raise Exception(f"Section {section} not found in {filename}")
    return config

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL", load_config()["uri"]))
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return {"error": "Database connection failed"}, 500

    try:
        data = pd.read_sql("SELECT * FROM esg_data", conn)
        sectors = data["sector"].unique()
        sector = request.form.get("sector", sectors[0])
        companies = data[data["sector"] == sector]["company"].unique()
        company = request.form.get("company", companies[0])

        # Risk score
        model = joblib.load("models/esg_risk_model.pkl")
        company_data = data[data["company"] == company]
        news_data = pd.read_sql("SELECT sentiment FROM news_social_data WHERE company = %s", conn, params=(company,))
        features = company_data[["carbon_emissions", "diversity_ratio", "governance_score"]].iloc[-1:]
        features["sentiment"] = news_data["sentiment"].mean() if not news_data.empty else 0
        features["volatility"] = pd.read_sql("SELECT volatility FROM financial_data WHERE company = %s", conn, params=(company,))["volatility"].mean()
        features["carbon_tax_rate"] = pd.read_sql("SELECT carbon_tax_rate FROM macro_data", conn)["carbon_tax_rate"].mean()
        risk_score = model.predict(features)[0]

        # Radar chart
        fig = px.line_polar(
            r=[company_data["carbon_emissions"].mean(), company_data["diversity_ratio"].mean() * 100, company_data["governance_score"].mean()],
            theta=["Environmental", "Social", "Governance"],
            line_close=True
        )
        radar_chart = pio.to_html(fig, full_html=False)

        # Sentiment trend
        news_data = pd.read_sql("SELECT date, sentiment FROM news_social_data WHERE company = %s", conn, params=(company,))
        fig = px.line(news_data, x="date", y="sentiment")
        trend_chart = pio.to_html(fig, full_html=False)

        conn.close()
        logger.info(f"Rendered dashboard for {company}")
        return render_template("index.html", sectors=sectors, companies=companies, sector=sector, company=company,
                              risk_score=risk_score, radar_chart=radar_chart, trend_chart=trend_chart)
    except Exception as e:
        logger.error(f"Failed to process dashboard: {e}")
        conn.close()
        return {"error": "Dashboard processing failed"}, 500

@app.route("/alerts/<company>")
def alerts(company):
    try:
        consumer = KafkaConsumer(
            os.getenv("KAFKA_TOPIC", "esg_topic"),
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256"),
            sasl_plain_username=os.getenv("KAFKA_SASL_USERNAME", ""),
            sasl_plain_password=os.getenv("KAFKA_SASL_PASSWORD", ""),
            group_id=f"esg_alerts_{company}"
        )
        logger.info(f"Connected to Redpanda for {company} alerts")
        alerts = []
        for message in consumer:
            news = json.loads(message.value.decode("utf-8"))
            if news["company"] == company and news["sentiment"] < -0.5:
                alerts.append(f"Alert: Negative sentiment for {company}: {news['text']}")
            if len(alerts) >= 5:  # Limit for demo
                break
        logger.info(f"Fetched {len(alerts)} alerts for {company}")
        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Failed to fetch alerts for {company}: {e}")
        return {"error": "Failed to fetch alerts"}, 500

@app.route("/report/<company>")
def generate_report(company):
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL", load_config()["uri"]))
        data = pd.read_sql("SELECT * FROM esg_data WHERE company = %s", conn, params=(company,))
        model = joblib.load("models/esg_risk_model.pkl")
        features = data[["carbon_emissions", "diversity_ratio", "governance_score"]].iloc[-1:]
        risk_score = model.predict(features)[0]

        pdf_file = f"reports/{company}_esg_report.pdf"
        os.makedirs("reports", exist_ok=True)
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(pdf_file, pagesize=letter)
        c.drawString(100, 750, f"ESG Risk Report for {company}")
        c.drawString(100, 730, f"Risk Score: {risk_score:.2f}")
        c.drawImage("static/shap_summary.png", 100, 500, width=400, height=200)
        c.save()

        conn.close()
        logger.info(f"Generated report for {company}")
        return send_file(pdf_file, as_attachment=True)
    except Exception as e:
        logger.error(f"Failed to generate report for {company}: {e}")
        if 'conn' in locals():
            conn.close()
        return {"error": "Report generation failed"}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))