# app.py
from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Load dataset + trained model
data = pd.read_csv("data/merged_data.csv")
model = joblib.load("models/esg_risk_model.pkl")

EXPECTED_FEATURES = list(model.named_steps["prep"].feature_names_in_)

companies = sorted(data["company"].unique())
sectors = sorted(data["sector"].unique())

def get_features_for_company(company: str):
    """Return latest feature row for a company as a DataFrame with correct columns."""
    row = data[data["company"] == company].sort_values("date").tail(1)
    if row.empty:
        return None
    return row[EXPECTED_FEATURES]

# --- Routes ---

@app.route("/")
def dashboard():
    company = request.args.get("company", companies[0])
    sector = request.args.get("sector", sectors[0])

    # KPIs
    avg_esg_score = round(data[data["sector"] == sector]["risk_score"].mean(), 2)
    total_carbon = round(data[data["sector"] == sector]["carbon_emissions"].sum(), 2)
    active_alerts = len(data[data["risk_score"] < 40])

    # Chart data (time series)
    filtered = data[data["company"] == company].sort_values("date")
    dates = filtered["date"].tolist()
    esg_scores = filtered["risk_score"].tolist()
    carbon_data = filtered["carbon_emissions"].tolist()

    # Predicted Risk Score
    features = get_features_for_company(company)
    predicted_risk = model.predict(features)[0] if features is not None else None

    # Portfolio-level distributions
    esg_distribution = (
        data.groupby("sector")["risk_score"].mean().to_dict()
        if "sector" in data.columns and "risk_score" in data.columns else {}
    )

    region_distribution = (
        data.groupby("region")["company"].nunique().to_dict()
        if "region" in data.columns else {}
    )

    # Company-level summary table
    df = (
        data.sort_values("date")
        .groupby("company")
        .tail(1)[["company", "sector", "region", "risk_score", "carbon_emissions", "governance_score", "diversity_ratio"]]
        .reset_index(drop=True)
        .rename(columns={
            "risk_score": "avg_esg_score",
            "diversity_ratio": "diversity_score"
        })
        .to_dict(orient="records")
    )

    return render_template(
        "dashboard.html",
        companies=companies,
        sectors=sectors,
        avg_esg_score=avg_esg_score,
        total_carbon=total_carbon,
        active_alerts=active_alerts,
        dates=dates,
        esg_scores=esg_scores,
        carbon_data=carbon_data,
        predicted_risk=round(predicted_risk, 2) if predicted_risk is not None else "N/A",
        selected_company=company,
        esg_distribution=esg_distribution,
        region_distribution=region_distribution,
        df=df
    )

@app.route("/compare")
def compare():
    c1 = request.args.get("c1", companies[0])
    c2 = request.args.get("c2", companies[1])

    f1 = get_features_for_company(c1)
    f2 = get_features_for_company(c2)

    pred1 = model.predict(f1)[0] if f1 is not None else None
    pred2 = model.predict(f2)[0] if f2 is not None else None

    return render_template(
        "compare.html",
        companies=companies,
        c1=c1, c2=c2,
        pred1=round(pred1, 2) if pred1 is not None else "N/A",
        pred2=round(pred2, 2) if pred2 is not None else "N/A"
    )

@app.route("/alerts")
def alerts():
    alerts = data[data["risk_score"] < 40][["company", "sector", "risk_score", "carbon_emissions"]]
    return render_template("alerts.html", tables=alerts.to_dict(orient="records"))

@app.route("/report")
def report():
    return render_template("report.html")

@app.route("/download_report")
def download_report():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Risk Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Total Companies: {len(companies)}", styles["Normal"]))
    story.append(Paragraph(f"Total Sectors: {len(sectors)}", styles["Normal"]))
    story.append(Paragraph(f"Average Risk Score: {round(data['risk_score'].mean(),2)}", styles["Normal"]))
    story.append(Paragraph(f"Alerts (Risk Score < 40): {len(data[data['risk_score']<40])}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Risk_Report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)
