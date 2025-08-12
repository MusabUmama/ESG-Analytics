from kafka import KafkaProducer
import pandas as pd
import json
import os
from configparser import ConfigParser

def load_config(filename="config/kafka_config.ini", section="kafka"):
    parser = ConfigParser()
    parser.read(filename)
    config = {k: v for k, v in parser.items(section)}
    config.update({
        "security_protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
        "sasl_mechanism": os.getenv("KAFKA_SASL_MECHANISM", "PLAIN"),
        "sasl_plain_username": os.getenv("KAFKA_SASL_USERNAME", "$ConnectionString"),
        "sasl_plain_password": os.getenv("KAFKA_SASL_PASSWORD", "")
    })
    return config

config = load_config()
producer = KafkaProducer(
    bootstrap_servers=config["bootstrap_servers"],
    security_protocol=config["security_protocol"],
    sasl_mechanism=config["sasl_mechanism"],
    sasl_plain_username=config["sasl_plain_username"],
    sasl_plain_password=config["sasl_plain_password"]
)
news_social_df = pd.read_csv("data/news_social_data.csv")
for _, row in news_social_df[news_social_df["date"] >= "2025-08-01"].iterrows():
    producer.send(config["topic"], value=json.dumps(row.to_dict()).encode("utf-8"))
producer.flush()