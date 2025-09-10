import base64
import urllib
from twilio.rest import Client
from settings import (
    TWILIO_AUTH_TOKEN, 
    TWILIO_WHATSAPP_FROM, 
    TWILIO_ACCOUNT_SID
)

# =========================
# Funções auxiliares
# =========================

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def _parse_twilio_body(event):
    """Parse do body do Twilio"""
    raw = event.get("body", "")
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    
    params = urllib.parse.parse_qs(raw, keep_blank_values=True)
    return {k: (v[0] if v else "") for k, v in params.items()}

def _send_whatsapp(to_number: str, message: str):
    """Envia mensagem via Twilio"""
    to = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
    msg = twilio_client.messages.create(
        from_=TWILIO_WHATSAPP_FROM,
        to=to,
        body=message
    )
    return msg.sid

