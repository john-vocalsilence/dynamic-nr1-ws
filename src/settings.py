import os
from typing import Dict

# OpenAI models / keys
MODEL_NAME: str = os.getenv("OPENAI_MODEL", "o3-2025-04-16")
SCREENING_MODEL: str = os.getenv("SCREENING_MODEL", "gpt-4.1-nano-2025-04-14")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")

# Twilio
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM", "")

# AWS / S3
S3_BUCKET: str = os.getenv("S3_BUCKET", "")

# Safety thresholds
SAFETY_CONFIDENCE_THRESHOLD: float = float(os.getenv("SAFETY_CONFIDENCE_THRESHOLD", "0.4"))

# Audio limits (seconds)
MAX_AUDIO_DURATION_MULTIPLE_CHOICE: int = int(os.getenv("MAX_AUDIO_DURATION_MULTIPLE_CHOICE", "15"))
MAX_AUDIO_DURATION_TEXT: int = int(os.getenv("MAX_AUDIO_DURATION_TEXT", "120"))
MAX_AUDIO_DURATION_LIKERT: int = int(os.getenv("MAX_AUDIO_DURATION_LIKERT", "15"))

# Database configuration
DB_CONFIG: Dict[str, object] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}

__all__ = [
    "MODEL_NAME",
    "SCREENING_MODEL",
    "OPENAI_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_WHATSAPP_FROM",
    "S3_BUCKET",
    "SAFETY_CONFIDENCE_THRESHOLD",
    "MAX_AUDIO_DURATION_MULTIPLE_CHOICE",
    "MAX_AUDIO_DURATION_TEXT",
    "MAX_AUDIO_DURATION_LIKERT",
    "DB_CONFIG",
]
