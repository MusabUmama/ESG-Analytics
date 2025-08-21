# src/kafka_prodcucer.py
from kafka import KafkaProducer
import pandas as pd
import json
import os
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config():
    config = {
        "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "topic": os.getenv("KAFKA_TOPIC", "esg_topic"),
        "security_protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
        "sasl_mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256"),
        "sasl_plain_username": os.getenv("KAFKA_SASL_USERNAME", ""),
        "sasl_plain_password": os.getenv("KAFKA_SASL_PASSWORD", "")
    }
    if not config["bootstrap_servers"] or not config["sasl_plain_username"] or not config["sasl_plain_password"]:
        logger.error("Missing Redpanda credentials (KAFKA_BOOTSTRAP_SERVERS, KAFKA_SASL_USERNAME, or KAFKA_SASL_PASSWORD)")
        raise ValueError("Redpanda credentials are required")
    return config

config = load_config()
try:
    producer = KafkaProducer(
        bootstrap_servers=config["bootstrap_servers"],
        security_protocol=config["security_protocol"],
        sasl_mechanism=config["sasl_mechanism"],
        sasl_plain_username=config["sasl_plain_username"],
        sasl_plain_password=config["sasl_plain_password"]
    )
    logger.info("Connected to Redpanda")
except Exception as e:
    logger.error(f"Failed to connect to Redpanda: {e}")
    raise

try:
    news_social_df = pd.read_csv("src/data/news_social_data.csv")
    for _, row in news_social_df[news_social_df["date"] >= "2025-08-01"].iterrows():
        producer.send(config["topic"], value=json.dumps(row.to_dict()).encode("utf-8"))
    producer.flush()
    logger.info("Streamed news_social_data.csv to esg_topic")
except Exception as e:
    logger.error(f"Failed to stream to Redpanda: {e}")
    raise