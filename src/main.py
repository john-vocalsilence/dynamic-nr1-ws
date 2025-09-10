import json
import logging
import time
import boto3
from agent import QuestionnaireStateMachine
from utils import _send_whatsapp, _parse_twilio_body

logger = logging.getLogger()
logger.setLevel(logging.INFO)

lambda_client = boto3.client("lambda")

def run(event, context):
    try:
        if "Records" in event:
                event = json.loads(event['Records'][0]['Sns']['Message'])
    
        # Modo background (execução assíncrona)
        if event.get("bg"):
            sender = event.get("bg_sender", "")
            user_message = event.get("bg_message", "")
            audio_info = event.get("bg_audio_info")  # Informações do áudio se houver
            
            try:
                machine = QuestionnaireStateMachine(sender)
                reply = machine.process_message(user_message, audio_info)
                
                # Suporta tanto string única quanto lista de mensagens
                if isinstance(reply, list):
                    for i, msg in enumerate(reply):
                        if msg.strip():
                            _send_whatsapp(sender, msg)
                            # Adiciona pequeno delay entre mensagens para garantir ordem
                            if i < len(reply) - 1:
                                time.sleep(0.5)  # 500ms entre mensagens
                elif reply.strip():
                    _send_whatsapp(sender, reply)
                
                return {"statusCode": 200, "body": "ok"}
                
            except Exception as e:
                print(f"[BG Error] {e}")
                try:
                    _send_whatsapp(sender, "Desculpe, houve um erro temporário. Pode repetir sua última mensagem?")
                except:
                    pass
                return {"statusCode": 200, "body": "error"}
        
        # Modo webhook (Twilio)
        method = event.get("httpMethod")
        
        if method == "POST":
            try:
                data = _parse_twilio_body(event)
                user_message = (data.get("Body") or "").strip()
                sender = (data.get("WaId") or data.get("From", "").replace("whatsapp:", "")).strip()
                
                if not sender:
                    return {"statusCode": 400, "body": "Missing sender"}
                
                # Verifica se há áudio na mensagem
                audio_info = None
                num_media = int(data.get("NumMedia", 0))
                
                if num_media > 0:
                    # Pega informações do primeiro áudio
                    media_url = data.get("MediaUrl0")
                    media_content_type = data.get("MediaContentType0", "audio/ogg")
                    
                    if media_url:
                        audio_info = {
                            "media_url": media_url,
                            "media_content_type": media_content_type,
                            "num_media": num_media
                        }
                        print(f"[Webhook] Áudio detectado: {media_url[:50]}..., tipo: {media_content_type}")
                        
                        # Se tem áudio, ignora o texto (geralmente vem vazio ou com emoji de microfone)
                        user_message = ""
                
                # Invoca execução assíncrona
                lambda_client.invoke(
                    FunctionName=context.invoked_function_arn,
                    InvocationType="Event",
                    Payload=json.dumps({
                        "bg": True,
                        "bg_sender": sender,
                        "bg_message": user_message,
                        "bg_audio_info": audio_info  # Passa informações do áudio
                    }).encode("utf-8")
                )
                
                # Resposta rápida ao Twilio
                twiml = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
                return {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/xml; charset=utf-8"},
                    "body": twiml
                }
                
            except Exception as e:
                print(f"[Webhook Error] {e}")
                return {"statusCode": 500, "body": "error"}
        
        return {"statusCode": 404, "body": "Not found"}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
