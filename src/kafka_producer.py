# src/kafka_prodcucer.py
import pandas as pd
import json
import os
import logging
from dotenv import load_dotenv
from confluent_kafka import Producer

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def delivery_report(err, msg):
    """Delivery callback to confirm message delivery"""
    if err is not None:
        logger.error(f"Delivery failed for record {msg.key()}: {err}")
    else:
        logger.info(f"Record successfully produced to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


def load_config():
    config = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "topic": os.getenv("KAFKA_TOPIC", "esg_topic"),
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
        "sasl.mechanisms": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256"),
        "sasl.username": os.getenv("KAFKA_SASL_USERNAME", ""),
        "sasl.password": os.getenv("KAFKA_SASL_PASSWORD", "")
    }
    if not config["bootstrap.servers"] or not config["sasl.username"] or not config["sasl.password"]:
        logger.error("Missing Redpanda credentials (KAFKA_BOOTSTRAP_SERVERS, KAFKA_SASL_USERNAME, or KAFKA_SASL_PASSWORD)")
        raise ValueError("Redpanda credentials are required")
    return config


def main():
    config = load_config()

    try:
        producer = Producer({
            "bootstrap.servers": config["bootstrap.servers"],
            "security.protocol": config["security.protocol"],
            "sasl.mechanisms": config["sasl.mechanisms"],
            "sasl.username": config["sasl.username"],
            "sasl.password": config["sasl.password"]
        })
        logger.info("Connected to Redpanda")
    except Exception as e:
        logger.error(f"Failed to connect to Redpanda: {e}")
        raise

    try:
        news_social_df = pd.read_csv("src/data/news_social_data.csv")

        for _, row in news_social_df[news_social_df["date"] >= "2025-08-01"].iterrows():
            message = json.dumps(row.to_dict())
            producer.produce(
                config["topic"],
                key=str(row.get("company", "unknown")), 
                value=message,
                callback=delivery_report
            )
            producer.poll(0)

        producer.flush()
        logger.info("Streamed news_social_data.csv to esg_topic")
    except Exception as e:
        logger.error(f"Failed to stream to Redpanda: {e}")
        raise


if __name__ == "__main__":
    main()
