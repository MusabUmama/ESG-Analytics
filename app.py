from flask import Flask, render_template, request, send_file
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.io as pio
import joblib
import os
from kafka import KafkaConsumer
import json

app = Flask(__name__)

def load_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_DATABASE", "esg_risk_db"),
        "user": os.getenv("DB_USER", "user"),
        "password": os.getenv("DB_PASSWORD", "password"),
        "port": os.getenv("DB_PORT", "5432"),
        "sslmode": "require"
    }

@app.route("/", methods=["GET", "POST"])
def index():
    conn = psycopg2.connect(**load_config())
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
    return render_template("index.html", sectors=sectors, companies=companies, sector=sector, company=company,
                           risk_score=risk_score, radar_chart=radar_chart, trend_chart=trend_chart)

@app.route("/alerts/<company>")
def alerts(company):
    consumer = KafkaConsumer(
        os.getenv("KAFKA_TOPIC", "esg_topic"),
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
        sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
        sasl_plain_username=os.getenv("KAFKA_SASL_USERNAME", "$ConnectionString"),
        sasl_plain_password=os.getenv("KAFKA_SASL_PASSWORD", "")
    )
    alerts = []
    for message in consumer:
        news = json.loads(message.value.decode("utf-8"))
        if news["company"] == company and news["sentiment"] < -0.5:
            alerts.append(f"Alert: Negative sentiment for {company}: {news['text']}")
        if len(alerts) >= 5:  # Limit for demo
            break
    return {"alerts": alerts}

@app.route("/report/<company>")
def generate_report(company):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    conn = psycopg2.connect(**load_config())
    data = pd.read_sql("SELECT * FROM esg_data WHERE company = %s", conn, params=(company,))
    model = joblib.load("models/esg_risk_model.pkl")
    features = data[["carbon_emissions", "diversity_ratio", "governance_score"]].iloc[-1:]
    risk_score = model.predict(features)[0]

    pdf_file = f"reports/{company}_esg_report.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    c.drawString(100, 750, f"ESG Risk Report for {company}")
    c.drawString(100, 730, f"Risk Score: {risk_score:.2f}")
    c.drawImage("static/shap_summary.png", 100, 500, width=400, height=200)
    c.save()

    conn.close()
    return send_file(pdf_file, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))