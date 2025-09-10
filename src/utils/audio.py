import requests
import tempfile
from typing import Dict, Any
from openai import OpenAI
from settings import (
    MAX_AUDIO_DURATION_MULTIPLE_CHOICE, 
    MAX_AUDIO_DURATION_LIKERT, 
    MAX_AUDIO_DURATION_TEXT, 
    OPENAI_API_KEY, 
    TWILIO_AUTH_TOKEN, 
    TWILIO_ACCOUNT_SID
)


# =========================
# Classe para Transcrição de Áudio
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

class AudioTranscriber:
    """Gerencia transcrição de áudios do WhatsApp"""
    
    @staticmethod
    def get_audio_duration_from_url(media_url: str, account_sid: str, auth_token: str) -> float:
        """Obtém duração aproximada do áudio sem baixar completamente"""
        try:
            # Faz requisição HEAD para obter tamanho do arquivo
            response = requests.head(
                media_url,
                auth=(account_sid, auth_token),
                timeout=5
            )
            
            # Estima duração baseado no tamanho (aproximação para áudio do WhatsApp)
            # WhatsApp usa OPUS codec ~6KB/s para voz
            content_length = int(response.headers.get('Content-Length', 0))
            if content_length > 0:
                estimated_duration = content_length / 6000  # 6KB por segundo
                return estimated_duration
            
            # Se não conseguir estimar, retorna duração máxima para forçar download
            return MAX_AUDIO_DURATION_TEXT
            
        except Exception as e:
            print(f"[Audio Duration Error] {e}")
            return MAX_AUDIO_DURATION_TEXT
    
    @staticmethod
    def download_audio(media_url: str, account_sid: str, auth_token: str) -> bytes:
        """Baixa arquivo de áudio do Twilio"""
        try:
            response = requests.get(
                media_url,
                auth=(account_sid, auth_token),
                timeout=30
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"[Audio Download Error] {e}")
            raise
    
    @staticmethod
    def transcribe_audio(audio_content: bytes, media_content_type: str = "audio/ogg") -> Dict[str, Any]:
        """Transcreve áudio usando OpenAI Whisper"""
        try:
            # Determina extensão baseado no content type
            extension = ".ogg"  # Padrão do WhatsApp
            if "mpeg" in media_content_type.lower():
                extension = ".mp3"
            elif "mp4" in media_content_type.lower():
                extension = ".mp4"
            elif "wav" in media_content_type.lower():
                extension = ".wav"
            
            # Cria arquivo temporário
            with tempfile.NamedTemporaryFile(suffix=extension, delete=True) as tmp_file:
                tmp_file.write(audio_content)
                tmp_file.flush()
                
                # Transcreve com Whisper
                with open(tmp_file.name, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="pt",  # Força português
                        response_format="verbose_json"  # Obtém mais informações
                    )
                
                return {
                    "success": True,
                    "text": transcript.text,
                    "duration": transcript.duration if hasattr(transcript, 'duration') else None,
                    "language": transcript.language if hasattr(transcript, 'language') else "pt"
                }
                
        except Exception as e:
            print(f"[Transcription Error] {e}")
            return {
                "success": False,
                "error": str(e),
                "text": None
            }
    
    @staticmethod
    def validate_audio_duration(duration: float, question_type: str) -> Dict[str, Any]:
        """Valida se duração do áudio está dentro dos limites"""
        max_duration = MAX_AUDIO_DURATION_TEXT  # Padrão
        
        if question_type == "multiple choice":
            max_duration = MAX_AUDIO_DURATION_MULTIPLE_CHOICE
        elif question_type == "likert":
            max_duration = MAX_AUDIO_DURATION_LIKERT
        elif question_type == "text":
            max_duration = MAX_AUDIO_DURATION_TEXT
        
        if duration > max_duration:
            return {
                "valid": False,
                "message": f"⚠️ Áudio muito longo ({duration:.0f}s). Para esta pergunta, envie áudios de até {max_duration}s.",
                "max_duration": max_duration
            }
        
        return {
            "valid": True,
            "max_duration": max_duration
        }
    
    @staticmethod
    def process_audio_message(media_url: str, media_content_type: str, 
                            question_type: str = "text") -> Dict[str, Any]:
        """Processa mensagem de áudio completa"""
        try:
            # Primeiro, estima duração sem baixar
            estimated_duration = AudioTranscriber.get_audio_duration_from_url(
                media_url, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
            )
            
            # Valida duração estimada
            duration_check = AudioTranscriber.validate_audio_duration(
                estimated_duration, question_type
            )
            
            # Se duração estimada já excede muito o limite, nem baixa
            if not duration_check["valid"] and estimated_duration > duration_check["max_duration"] * 2:
                return {
                    "success": False,
                    "message": duration_check["message"],
                    "transcription": None
                }
            
            # Baixa o áudio
            print(f"[Audio] Baixando áudio de {media_url[:50]}...")
            audio_content = AudioTranscriber.download_audio(
                media_url, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
            )
            
            # Transcreve
            print(f"[Audio] Transcrevendo áudio ({len(audio_content)} bytes)...")
            transcription_result = AudioTranscriber.transcribe_audio(
                audio_content, media_content_type
            )
            
            if not transcription_result["success"]:
                return {
                    "success": False,
                    "message": "❌ Não consegui transcrever o áudio. Por favor, envie uma mensagem de texto ou tente novamente.",
                    "transcription": None,
                    "error": transcription_result.get("error")
                }
            
            # Valida duração real se disponível
            if transcription_result.get("duration"):
                real_duration = transcription_result["duration"]
                duration_check = AudioTranscriber.validate_audio_duration(
                    real_duration, question_type
                )
                
                if not duration_check["valid"]:
                    return {
                        "success": False,
                        "message": duration_check["message"],
                        "transcription": transcription_result["text"],
                        "duration": real_duration
                    }
            
            # Sucesso
            return {
                "success": True,
                "transcription": transcription_result["text"],
                "duration": transcription_result.get("duration"),
                "language": transcription_result.get("language", "pt")
            }
            
        except Exception as e:
            print(f"[Audio Processing Error] {e}")
            return {
                "success": False,
                "message": "❌ Erro ao processar áudio. Por favor, envie uma mensagem de texto.",
                "transcription": None,
                "error": str(e)
            }


