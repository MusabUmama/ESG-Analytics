# Predictive ESG Analytics System

This project implements a predictive system to assess Environmental, Social, and Governance (ESG) risk scores for companies using machine learning. It integrates data from various sources, preprocesses it, trains a model, streams results to Redpanda Cloud, and provides a Flask-based web interface.

![Dashboard](https://github.com/MusabUmama/ESG-Analytics/blob/main/Dashboard.png)

## Features

- Generates synthetic ESG, financial, and macro-economic data.
- Preprocesses data into a merged dataset and stores it in Supabase.
- Trains a model to predict ESG risk scores.
- Streams data to Redpanda Cloud for real-time processing.
- Provides a web dashboard, alerts, and company reports via Flask.

## Prerequisites

- Python 3.11+
- Git
- Internet connection (for package installation and API calls)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ESG-Risk-Scoring-System.git
cd ESG-Risk-Scoring-System
```

### 2. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install --upgrade pip together psycopg2-binary kafka-python shap xgboost
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory with the following:

```
TOGETHER_API_KEY=tgp_v1_Q7hE8cwG0rwmi7OjxNhepA9BCBVLmTEqkyYCsIxxql4
DATABASE_URL=postgresql://postgres:<supabase-password>@db.<project-id>.supabase.co:5432/postgres
KAFKA_BOOTSTRAP_SERVERS=<cluster>.cloud.redpanda.com:9092
KAFKA_TOPIC=esg_topic
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=SCRAM-SHA-256
KAFKA_SASL_USERNAME=<redpanda-username>
KAFKA_SASL_PASSWORD=<redpanda-password>
```

- Replace `<supabase-password>`, `<project-id>`, `<cluster>`, `<redpanda-username>`, and `<redpanda-password>` with your actual credentials.

### 4. Verify Setup

Test environment variable loading:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('DATABASE_URL')); print(os.getenv('KAFKA_BOOTSTRAP_SERVERS'))"
```

## Usage

### 1. Generate Data

Run the data generation script:

```bash
python src/generate_data.py > generate_data_log.txt
```

- Output: CSV files in `data/` (e.g., `companies.csv`, `merged_data.csv`).

### 2. Preprocess Data

Run the preprocessing script:

```bash
python src/preprocess_data.py > preprocess_data_log.txt
```

- Output: `data/merged_data.csv` and data inserted into Supabase `esg_data` table.
- Verify: `psql "<DATABASE_URL>" -c "SELECT * FROM esg_data LIMIT 5;"`.

### 3. Train Model

Run the model training script:

```bash
python src/training_esh.ipynb
```

- Output: `models/esg_risk_model.pkl`, `static/shap_summary.png`, `static/shap_summary_bar.png`.
- Verify: `dir models\esg_risk_model.pkl` and `dir static\`.

### 4. Stream to Redpanda

Run the Kafka producer:

```bash
python src/kafka_producer.py > kafka_producer_log.txt
```

- Verify: Check Redpanda Dashboard (Topics > `esg_topic` > Messages).

### 5. Run Web Application

Start the Flask app:

```bash
python src/app.py > app_log.txt
```

- Access:
  - Dashboard: `http://localhost:5000`
  - Alerts: `curl http://localhost:5000/alerts/AAPL`
  - Report: `http://localhost:5000/report/AAPL`
- Verify: Check `app_log.txt` for errors.

### Supabase Setup

- Create a Supabase project and obtain `DATABASE_URL`.
- Ensure the `esg_data` table is created during preprocessing.

### Redpanda Cloud Setup

- Create a Redpanda Cloud cluster and obtain `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC`, and credentials.
- Verify topic `esg_topic` exists.

## Project Structure

```
ESG-Risk-Scoring-System/
├── data/              # Generated CSV files
├── models/            # Trained model files
├── static/            # plots
├── templates/         # HTML templates for Flask
├── src/              # Source code
│   ├── generate_data.py
│   ├── preprocess_data.py
│   ├── training_esg.ipynb
│   ├── kafka_producer.py
│   └── app.py
├── .env              # Environment variables
├── requirements.txt  # Python dependencies
└── README.md         
```

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.
