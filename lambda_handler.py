# lambda_function.py
from openai import OpenAI
from datetime import datetime, timedelta
import os
import urllib
import json
import boto3
import base64
import unicodedata
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
from twilio.rest import Client
from psycopg2 import pool
from contextlib import contextmanager
import atexit
import re
import time
import requests
import tempfile

# =========================
# Configuração
# =========================
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-2025-04-16")
SCREENING_MODEL = os.getenv("SCREENING_MODEL", "gpt-4.1-nano-2025-04-14")  # Modelo para triagem inicial
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_FROM = os.environ["TWILIO_WHATSAPP_FROM"]
S3_BUCKET = os.environ["S3_BUCKET"]

# Limiar de confiança para acionar protocolo detalhado
SAFETY_CONFIDENCE_THRESHOLD = 0.4

# Configurações de áudio
MAX_AUDIO_DURATION_MULTIPLE_CHOICE = 15  # segundos
MAX_AUDIO_DURATION_TEXT = 120  # segundos (2 minutos)
MAX_AUDIO_DURATION_LIKERT = 15  # segundos

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

# Clientes
client = OpenAI(api_key=OPENAI_API_KEY)
s3 = boto3.client("s3")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
lambda_client = boto3.client("lambda")

# Pool de conexões
db_pool = None

# Cache do questionário
_questionnaire_cache = None

# =========================
# Estados da Máquina
# =========================
class State(Enum):
    WELCOME = "welcome"
    CONSENT = "consent"
    PHASE1_QUESTIONS = "phase1_questions"
    ASSESSMENT = "assessment"
    FOLLOWUP_INTRO = "followup_intro"
    FOLLOWUP_QUESTIONS = "followup_questions"
    ORIGIN_INTRO = "origin_intro"
    ORIGIN_QUESTIONS = "origin_questions"
    COMPLETION = "completion"
    EMERGENCY = "emergency"
    RESET = "reset"

# =========================
# Constantes de Follow-up
# =========================
FOLLOWUP_QUESTIONS = [
    {"id": "A1", "question": "Já tive afastamento do trabalho por alguma causa psicossocial?", "options": ["Sim", "Não", "Prefiro não responder"]},
    {"id": "A2", "question": "Nas últimas duas semanas, você se sentiu para baixo, deprimido ou sem esperanças?", "options": ["Sim", "Não", "Prefiro não responder"]},
    {"id": "A3", "question": "Você perdeu o interesse ou o prazer em fazer coisas que normalmente gosta?", "options": ["Sim", "Não", "Prefiro não responder"]},
    {"id": "A4", "question": "Nas últimas duas semanas, você se sentiu nervoso, ansioso ou tenso?", "options": ["Sim", "Não", "Prefiro não responder"]},
    {"id": "A5", "question": "Você teve dificuldade em parar ou controlar as preocupações?", "options": ["Sim", "Não", "Prefiro não responder"]},
    {"id": "A6", "question": "Você tem se sentido tão agitado que é difícil ficar parado ou relaxar?", "options": ["Sim", "Não", "Prefiro não responder"]},
]

# NOVAS PERGUNTAS DE ORIGEM - Apenas 2 perguntas
ORIGIN_QUESTIONS = [
    {
        "id": "O1", 
        "question": "Sabemos que coisas do trabalho e da vida pessoal podem se misturar. Para agir melhor, de onde vem essa situação?",
        "type": "multiple choice",
        "options": ["Do ambiente de trabalho", "Da vida pessoal", "Dos dois", "Prefiro não responder"]
    },
    {
        "id": "O2", 
        "question": "É importante saber se a empresa tem condições de agir sobre o que você trouxe. Na sua opinião, a empresa poderia fazer algo para melhorar essa situação?",
        "type": "text"
    }
]

# MAPEAMENTO DE DIMENSÕES PARA DESCRIÇÕES DETALHADAS
DIMENSION_DESCRIPTIONS = {
    "Qualidade do sono e disposição": "na qualidade do sono e disposição e indícios de fadiga/insônia",
    "Ânimo e motivação": "no ânimo e motivação e indícios de falta de ânimo positivo e satisfação no trabalho",
    "Estresse e ansiedade": "em lidar com demandas sem estresse e manter rendimento",
    "Equilíbrio vida-trabalho": "no equilíbrio trabalho e descanso",
    "Exigências de tempo no trabalho": "nas exigências de tempo no trabalho, indícios de pressão contínua e falta de tempo suficiente para a vida pessoal"
}

# =========================
# Database
# =========================
def init_db_pool():
    global db_pool
    if db_pool is None:
        try:
            db_pool = pool.ThreadedConnectionPool(
                1, 5,
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"],
                dbname=DB_CONFIG["dbname"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                connect_timeout=15
            )
            print(f"[DB Pool] Created successfully")
        except Exception as e:
            print(f"[DB Pool Error] {e}")
            raise

init_db_pool()

def close_db_pool():
    global db_pool
    if db_pool:
        db_pool.closeall()

atexit.register(close_db_pool)

@contextmanager
def get_db_connection():
    global db_pool
    conn = None
    try:
        if db_pool is None:
            init_db_pool()
        conn = db_pool.getconn()
        if conn:
            yield conn
            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB Error] {e}")
        raise
    finally:
        if conn and db_pool:
            db_pool.putconn(conn)

# =========================
# Gestão de Crise com LLM
# =========================
class CrisisManager:
    """Gerencia conversas durante crises de saúde mental"""
    
    def __init__(self, sender_id: str, load_existing: bool = True):
        self.sender_id = sender_id
        self.crisis_history = []
        self.crisis_type = None
        self.safety_score = 0
        self.interaction_count = 0
        # Só carrega estado existente se solicitado
        if load_existing:
            self.load_crisis_state()
    
    def load_crisis_state(self):
        """Carrega estado da crise do banco de dados"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT crisis_history, crisis_type, safety_score, interaction_count
                        FROM crisis_state
                        WHERE sender_id = %s AND active = true
                        ORDER BY created_at DESC LIMIT 1
                    """, (self.sender_id,))
                    
                    result = cur.fetchone()
                    if result:
                        self.crisis_history = json.loads(result[0]) if result[0] else []
                        self.crisis_type = result[1]
                        self.safety_score = result[2] or 0
                        self.interaction_count = result[3] or 0
                        print(f"[Crisis State] Estado existente carregado: tipo={self.crisis_type}, interações={self.interaction_count}, histórico={len(self.crisis_history)} mensagens")
                        
                        # Verifica se crisis_type é válido
                        if not self.crisis_type:
                            print(f"[Crisis State] AVISO: crisis_type está None/vazio no banco de dados! Definindo como 'unknown'")
                            self.crisis_type = 'unknown'  # Define um padrão
                            # Atualiza no banco com o tipo corrigido
                            cur.execute("""
                                UPDATE crisis_state 
                                SET crisis_type = %s, updated_at = %s
                                WHERE sender_id = %s AND active = true
                            """, ('unknown', datetime.utcnow(), self.sender_id))
                    else:
                        # Não é erro - apenas não há estado anterior (primeira crise ou após reset)
                        print(f"[Crisis State] Novo estado de crise será criado para {self.sender_id}")
        except Exception as e:
            print(f"[Crisis State Load Error] {e}")
            # Em caso de erro, mantém valores padrão já inicializados
    
    def save_crisis_state(self):
        """Salva estado da crise no banco de dados"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO crisis_state 
                        (sender_id, crisis_history, crisis_type, safety_score, 
                         interaction_count, active, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (sender_id, active) 
                        WHERE active = true
                        DO UPDATE SET
                            crisis_history = EXCLUDED.crisis_history,
                            crisis_type = EXCLUDED.crisis_type,
                            safety_score = EXCLUDED.safety_score,
                            interaction_count = EXCLUDED.interaction_count,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        self.sender_id,
                        json.dumps(self.crisis_history, ensure_ascii=False),
                        self.crisis_type,
                        self.safety_score,
                        self.interaction_count,
                        True,
                        datetime.utcnow()
                    ))
        except Exception as e:
            print(f"[Crisis State Save Error] {e}")
    
    def end_crisis(self, reason: str):
        """Finaliza a crise e marca como inativa"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE crisis_state 
                    SET active = false, 
                        resolution_reason = %s,
                        resolved_at = %s
                    WHERE sender_id = %s AND active = true
                """, (reason, datetime.utcnow(), self.sender_id))
    
    def get_crisis_prompt(self) -> str:
        """Gera prompt específico para o tipo de crise"""
        base_prompt = f"""Você é um assistente de saúde mental treinado, conduzindo uma conversa de suporte durante uma crise.

CONTEXTO DA CRISE:
- Tipo de risco detectado: {self.crisis_type}
- Número de interações até agora: {self.interaction_count}
- Score de segurança atual (0-10, onde 10 é seguro): {self.safety_score}

HISTÓRICO RECENTE DA CONVERSA:
{self._format_history()}

PROTOCOLO PARA {(self.crisis_type or 'UNKNOWN').upper()}:
{self._get_protocol_for_type()}

SUAS RESPONSABILIDADES:
1. A conversa deve ser empática e não-julgamental
2. Avaliar continuamente o estado emocional do usuário
3. Oferecer recursos de emergência quando apropriado (sem ser repetitivo)
4. Conduzir a conversa até que o usuário esteja estabilizado
5. NUNCA minimizar os sentimentos do usuário
6. SEMPRE validar as emoções antes de oferecer soluções

INSTRUÇÃO ESPECIAL - RETOMADA DO QUESTIONÁRIO:
- Se o usuário expressar QUALQUER uma dessas situações:
  * "estou melhor" / "já estou melhor" / "me sinto melhor"
  * "quero continuar" / "desejo continuar" / "continuar o questionário"
  * "voltar ao questionário" / "retomar o questionário"
  * "já passou" / "tá tudo bem" / "estou bem"
- E você avaliar que ele está minimamente estável (não precisa estar 100% perfeito)
- VOCÊ DEVE OBRIGATORIAMENTE:
  1. Escrever uma mensagem de acolhimento e confirmação
  2. TERMINAR sua resposta EXATAMENTE com: [RETOMAR_QUESTIONARIO]
  
EXEMPLO OBRIGATÓRIO de resposta quando usuário quer continuar:
"Que bom que você está se sentindo melhor! Fico feliz em saber que quer continuar. Vamos retomar o questionário de onde paramos. [RETOMAR_QUESTIONARIO]"

CRITÉRIOS FLEXÍVEIS PARA RETOMADA:
- Usuário expressou melhora OU desejo de continuar (não precisa ser os dois)
- Não há sinais de risco IMINENTE (pode haver algum desconforto residual)
- Usuário parece capaz de responder perguntas simples
- Se o usuário INSISTE em continuar, PERMITA (mesmo que você tenha dúvidas)

IMPORTANTE:
- Use a frase [RETOMAR_QUESTIONARIO] apenas quando tiver ABSOLUTA certeza de que é seguro
- Se tiver qualquer dúvida, continue a conversa de apoio
- Número atual de mensagens na conversa: {self.interaction_count}
- Mantenha um tom caloroso, humano e acolhedor
- Use linguagem simples e acessível
- Responda em português brasileiro

Responda à última mensagem do usuário de forma empática e helpful."""

        return base_prompt
    
    def _format_history(self) -> str:
        """Formata histórico para o prompt"""
        if not self.crisis_history:
            return "Início da conversa de suporte"
        
        formatted = []
        for entry in self.crisis_history[-5:]:  # Últimas 5 mensagens
            role = "Usuário" if entry['role'] == 'user' else "Assistente"
            formatted.append(f"{role}: {entry['content']}")
        
        return "\n".join(formatted)
    
    def _get_protocol_for_type(self) -> str:
        """Retorna protocolo específico por tipo de crise"""
        protocols = {
            'suicide': """
⚠️ Sinto muito pelo que você está vivendo. Sua vida é valiosa.
👉 Se você está em perigo imediato, ligue 190.
👉 Você também pode ligar agora para o 188 (CVV – Centro de Valorização da Vida). É gratuito, sigiloso e funciona 24h.
👉 Se puder, procure também o RH ou o canal de apoio da sua empresa, que pode indicar ajuda próxima.
👉 Se quiser, você pode compartilhar como acredita que a empresa pode ajudar nessa situação. Podemos fazer sua voz ser registrada de forma segura.
A Vocal Silence não substitui serviços médicos ou de emergência. Procure ajuda especializada sempre que precisar.
""",
            'violence': """
⚠️ Entendemos a seriedade do que você compartilhou.
Se você está em risco ou pensa em machucar alguém, é muito importante buscar ajuda imediata.
👉 Em situações de sofrimento intenso, você também pode ligar para o 188 (CVV – Centro de Valorização da Vida), disponível 24 horas por dia, gratuitamente.
👉 Além disso, você pode procurar o RH ou o canal de apoio da sua empresa, que poderá orientar sobre medidas de proteção e acolhimento.
👉 Se quiser, você pode compartilhar como acredita que a empresa pode ajudar nessa situação. Podemos fazer sua voz ser registrada de forma segura.
A Vocal Silence não substitui serviços médicos ou de emergência. Procure ajuda especializada sempre que precisar.
""",
            'substance': """
⚠️ Obrigado por compartilhar algo tão sensível.
Sabemos que o uso de substâncias pode ser difícil de lidar e não estamos aqui para julgar, mas para ouvir.
👉 Se quiser, você pode compartilhar como acredita que a empresa pode ajudar nessa situação. Podemos fazer sua voz ser registrada de forma segura.
👉 Se você sente que precisa de apoio, pode procurar serviços especializados como o CAPS AD (Centro de Atenção Psicossocial Álcool e Drogas) na sua região, ou grupos de apoio como AA (Alcoólicos Anônimos) e NA (Narcóticos Anônimos).
👉 O processo de mudança é desafiador, e recaídas fazem parte da recuperação – não significam fracasso.
A Vocal Silence não substitui acompanhamento médico ou terapêutico. Procure ajuda especializada sempre que precisar.
""",
            'psychosis': """
⚠️ Obrigado por compartilhar sua experiência.
Percebemos que você pode estar passando por um momento delicado e é muito importante procurar ajuda profissional o quanto antes.
👉 Se houver risco imediato para você ou para outras pessoas, ligue 190.
👉 Também é fundamental buscar atendimento médico ou em um CAPS (Centro de Atenção Psicossocial) na sua região, que conta com equipes preparadas para acolher situações como essa.
👉 Se sentir confortável, você pode compartilhar como a empresa pode apoiar nesse contexto. Podemos registrar sua voz de forma segura.
A Vocal Silence não substitui acompanhamento médico ou terapêutico, mas reforçamos a importância de procurar ajuda especializada.
""",
            'help_request': """
⚠️ Percebo que você está passando por um momento difícil e precisa de ajuda.
👉 Você pode ligar para o 188 (CVV – Centro de Valorização da Vida). É gratuito, sigiloso e funciona 24h.
👉 Em emergências, ligue 190 ou 192 (SAMU).
👉 Se puder, procure também o RH ou o canal de apoio da sua empresa.
👉 Como você está se sentindo agora? Estou aqui para ouvir e apoiar você.
A Vocal Silence não substitui serviços médicos ou de emergência, mas estamos aqui para acolher você neste momento.
"""
        }
        
        return protocols.get(self.crisis_type or 'help_request', protocols.get('help_request', protocols['suicide']))
    
    def evaluate_safety(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Avalia o nível de segurança após cada interação"""
        try:
            # Verifica se usuário expressou melhora explicitamente
            improvement_phrases = [
                'estou melhor', 'tô melhor', 'to melhor', 'me sinto melhor',
                'já passou', 'passou já', 'tá tudo bem', 'ta tudo bem',
                'estou bem', 'to bem', 'tô bem', 'me sinto bem',
                'quero continuar', 'continuar o questionário', 
                'voltar ao questionário', 'seguir com o questionário'
            ]
            
            # Frases que indicam melhora real no estado emocional
            emotional_improvement_phrases = [
                'estou melhor', 'me sinto melhor', 'tô melhor', 'to melhor',
                'estou bem', 'me sinto bem', 'tô bem', 'to bem',
                'já passou', 'passou já', 'tá tudo bem', 'ta tudo bem',
                'me acalmei', 'tô mais calmo', 'to mais calma'
            ]
            
            user_msg_lower = user_message.lower()
            explicit_improvement = any(phrase in user_msg_lower for phrase in improvement_phrases)
            emotional_improvement = any(phrase in user_msg_lower for phrase in emotional_improvement_phrases)
            
            eval_prompt = f"""Avalie a segurança desta conversa de crise.

Tipo de crise: {self.crisis_type}
Mensagem do usuário: "{user_message}"
Resposta do assistente: "{assistant_response}"
Score de segurança anterior: {self.safety_score}/10
Usuário expressou melhora explicitamente: {explicit_improvement}
Usuário expressou melhora emocional real: {emotional_improvement}

Analise e responda APENAS em JSON:
{{
  "safety_score": 0-10 (10 = totalmente seguro),
  "risk_level": "critical" | "high" | "medium" | "low",
  "can_resume_questionnaire": true/false,
  "reasoning": "breve explicação",
  "user_expressed_improvement": true/false,
  "emotional_improvement_detected": true/false,
  "specific_improvements": ["lista de melhorias observadas"]
}}

Considere:
- O usuário expressou estar melhor? (detectado: {explicit_improvement})
- Há melhora emocional real? (detectado: {emotional_improvement})
- Há sinais de estabilização emocional?
- Os riscos diminuíram?
- É seguro retomar atividades normais?

CRITÉRIOS FLEXÍVEIS:
- Se usuário disse estar melhor/bem E score anterior >= 3: pode retomar
- Se há melhora emocional clara E score anterior >= 4: pode retomar  
- Se usuário insiste em continuar E mostra estabilidade: pode retomar após confirmação"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": eval_prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"[Safety Evaluation Error] {e}")
            return {
                "safety_score": self.safety_score,
                "risk_level": "high",
                "can_resume_questionnaire": False,
                "reasoning": "Erro na avaliação",
                "user_expressed_improvement": False,
                "specific_improvements": []
            }
    
    def handle_crisis_conversation(self, user_message: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Gerencia conversa durante crise - LLM conduz completamente a conversa
        Retorna: (resposta, pode_retomar_questionario, metadata)
        """
        self.interaction_count += 1
        
        # Adiciona mensagem ao histórico
        self.crisis_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Gera resposta com LLM
        try:
            prompt = self.get_crisis_prompt()
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_completion_tokens=600
            )
            
            assistant_response = response.choices[0].message.content
            print(f"[Crisis Manager] Resposta do LLM (primeiros 200 caracteres): {assistant_response[:200]}")
            
            # Adiciona resposta ao histórico
            self.crisis_history.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Verifica se a LLM sinalizou para retomar o questionário (múltiplas variações)
            resume_signals = [
                "[RETOMAR_QUESTIONARIO]",
                "[RETOMAR QUESTIONARIO]", 
                "[RETOMAR_QUESTIONÁRIO]",
                "[RETOMAR QUESTIONÁRIO]"
            ]
            
            can_resume = False
            for signal in resume_signals:
                if signal in assistant_response:
                    can_resume = True
                    # Remove o sinal da resposta mas garante que há conteúdo
                    assistant_response = assistant_response.replace(signal, "").strip()
                    print(f"[Crisis Manager] Sinal de retomada detectado: {signal}")
                    print(f"[Crisis Manager] Resposta após remover sinal: '{assistant_response}'")
                    
                    # Se a resposta ficou vazia após remover o sinal, adiciona mensagem padrão
                    if not assistant_response:
                        assistant_response = "Que bom que você está se sentindo melhor! Vamos retomar o questionário de onde paramos."
                        print(f"[Crisis Manager] Resposta estava vazia, usando mensagem padrão")
                    break
            
            if can_resume:
                # Finaliza a crise
                self.end_crisis(f"LLM avaliou que usuário está pronto para retomar. Interações: {self.interaction_count}")
                print(f"[Crisis Manager] LLM sinalizou retomada após {self.interaction_count} interações")
            
            # Mecanismo de segurança: se muitas interações sem retomada, oferece opção
            elif self.interaction_count >= 10:
                assistant_response += f"\n\n💡 Nota: Já conversamos bastante ({self.interaction_count} mensagens). Se você se sente melhor e quer continuar o questionário, me avise diretamente."
            
            # Atualiza score de segurança baseado na progressão da conversa
            if can_resume:
                self.safety_score = 8  # Score alto se LLM aprovou retomada
            else:
                # Incrementa gradualmente o score conforme a conversa progride
                self.safety_score = min(self.safety_score + 0.5, 6)
            
            # Salva estado
            self.save_crisis_state()
            
            # Adiciona recursos de emergência periodicamente (a cada 4 mensagens)
            if self.interaction_count % 4 == 0 and not can_resume:
                assistant_response += "\n\n📞 Lembre-se: CVV 188 (24h) | SAMU 192"
            
            metadata = {
                "crisis_type": self.crisis_type,
                "interaction_count": self.interaction_count,
                "safety_score": self.safety_score,
                "can_resume": can_resume,
                "llm_managed": True,
                "resume_signal_detected": can_resume
            }
            
            return assistant_response, can_resume, metadata
            
        except Exception as e:
            print(f"[Crisis Conversation Error] {e}")
            return (
                "Estou aqui para te apoiar. Como você está se sentindo agora? "
                "Lembre-se que há ajuda disponível: CVV 188 (24h) | SAMU 192",
                False,
                {"error": str(e)}
            )

# =========================
# Protocolo de Segurança Aprimorado
# =========================
class SafetyProtocol:
    EMERGENCY_KEYWORDS = {
        'suicide': ['suicídio', 'suicidio', 'me matar', 'tirar minha vida', 'não aguento mais', 
                   'nao aguento mais', 'acabar com tudo', 'desistir de viver', 'me cortar', 
                   'me machucar', 'automutilação', 'automutilacao'],
        'violence': ['matar alguém', 'matar alguem', 'machucar', 'violência', 'violencia', 
                     'agredir', 'ferir', 'atacar', 'vingança', 'vinganca'],
        'substance': ['overdose', 'drogas', 'álcool', 'alcool', 'vício', 'vicio', 
                     'dependência', 'dependencia', 'abuso de substância'],
        'psychosis': ['vozes', 'alucinação', 'alucinacao', 'perseguição', 'perseguicao',
                     'complô', 'complo', 'delírio', 'delirio', 'paranoia'],
        'help_request': ['me sentindo mal', 'sentindo mal', 'preciso de ajuda', 'preciso ajuda',
                        'contato de alguém', 'contato de alguem', 'contato para ajudar',
                        'alguém para me ajudar', 'alguem para me ajudar']
    }
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normaliza texto removendo acentos e convertendo para minúsculo"""
        nfkd = unicodedata.normalize('NFKD', text.lower())
        return ''.join([c for c in nfkd if not unicodedata.combining(c)])
    
    @classmethod
    def quick_check(cls, message: str) -> Dict[str, Any]:
        """Verificação rápida por palavras-chave (usado como fallback)"""
        normalized_msg = cls.normalize_text(message)
        
        for emergency_type, keywords in cls.EMERGENCY_KEYWORDS.items():
            for keyword in keywords:
                if cls.normalize_text(keyword) in normalized_msg:
                    return {'detected': True, 'type': emergency_type, 'keyword': keyword}
        
        return {'detected': False}
    
    @classmethod
    def llm_screening(cls, message: str) -> Dict[str, Any]:
        """Triagem inicial com modelo leve - SEMPRE executada"""
        try:
            prompt = f"""Analise esta mensagem e identifique possíveis riscos de segurança em saúde mental.

Mensagem: "{message}"

Categorias de risco:
1. suicide: menções a suicídio, automutilação, desistir de viver
2. violence: ameaças de violência contra outros, homicídio
3. substance: abuso de substâncias, overdose, dependência química
4. psychosis: sinais de psicose, alucinações, delírios, paranoia
5. help_request: pedidos de ajuda, contatos de emergência, menções a se sentir mal e precisar de apoio
6. none: nenhum risco detectado

IMPORTANTE: 
- "Estou me sentindo mal" + pedido de ajuda/contato = help_request
- "Preciso de ajuda" ou "tem contato de alguém" = help_request
- Qualquer pedido de contato ou ajuda profissional = help_request

Responda EXATAMENTE neste formato JSON:
{{
  "has_risk": true/false,
  "type": "suicide" ou "violence" ou "substance" ou "psychosis" ou "help_request" ou "none",
  "confidence": 0.0 a 1.0,
  "reasoning": "breve explicação em português"
}}

Seja conservador - na dúvida, marque como risco."""

            response = client.chat.completions.create(
                model=SCREENING_MODEL,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "has_risk": result.get("has_risk", False),
                "type": result.get("type", "none"),
                "confidence": result.get("confidence", 0),
                "reasoning": result.get("reasoning", ""),
                "screening_model": SCREENING_MODEL
            }
            
        except Exception as e:
            print(f"[Safety Screening Error] {e}")
            # Fallback para quick_check
            print("[Safety] Usando quick_check como fallback")
            quick_result = cls.quick_check(message)
            if quick_result['detected']:
                return {
                    "has_risk": True,
                    "type": quick_result['type'],
                    "confidence": 0.8,
                    "reasoning": f"Detectado por palavra-chave: {quick_result['keyword']}",
                    "screening_model": "quick_check"
                }
            return {
                "has_risk": False,
                "type": "none",
                "confidence": 0,
                "reasoning": "Sem riscos detectados",
                "screening_model": "quick_check"
            }
    
    @classmethod
    def llm_detailed_check(cls, message: str, initial_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Verificação detalhada com modelo avançado quando risco é detectado"""
        try:
            risk_type = initial_assessment.get('type', 'unknown')
            
            prompt = f"""Você é um especialista em saúde mental analisando uma mensagem de risco.

Mensagem do usuário: "{message}"

Avaliação inicial indicou possível risco de: {risk_type}
Razão: {initial_assessment.get('reasoning', 'N/A')}

Faça uma análise DETALHADA e responda em JSON:
{{
  "is_emergency": true/false,
  "type": "{risk_type}" ou outro tipo se mais apropriado,
  "severity": "low" ou "medium" ou "high" ou "critical",
  "confidence": 0.0 a 1.0,
  "detailed_analysis": "análise detalhada em português",
  "recommended_action": "ação recomendada",
  "initial_safety_score": 0-10 (10 = seguro),
  "requires_immediate_intervention": true/false
}}

Considere:
- Gravidade e iminência do risco
- Contexto e nuances da mensagem
- Necessidade de intervenção imediata
- Tom apropriado para a resposta"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            result["detailed_check_model"] = MODEL_NAME
            return result
            
        except Exception as e:
            print(f"[Safety Detailed Check Error] {e}")
            # Em caso de erro, assume emergência por precaução
            return {
                "is_emergency": True,
                "type": initial_assessment.get('type', 'unknown'),
                "severity": "high",
                "confidence": 0.7,
                "detailed_analysis": "Erro na análise detalhada - assumindo emergência por precaução",
                "recommended_action": "Iniciar protocolo de suporte",
                "initial_safety_score": 3,
                "requires_immediate_intervention": True,
                "detailed_check_model": "error"
            }

# =========================
# Parser de Respostas (mantido igual)
# =========================
class ResponseParser:
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normaliza texto para comparação"""
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return ''.join([c for c in nfkd if not unicodedata.combining(c)])
    
    @classmethod
    def parse_multiple_choice(cls, message: str, options: list) -> Dict[str, Any]:
        """Parse de múltipla escolha"""
        message = message.strip()
        
        # Mapeia emojis de números para índices
        number_emoji_map = {
            '1️⃣': 0, '2️⃣': 1, '3️⃣': 2, '4️⃣': 3, '5️⃣': 4,
            '6️⃣': 5, '7️⃣': 6, '8️⃣': 7, '9️⃣': 8
        }
        
        # Tenta emoji de número primeiro
        if message in number_emoji_map:
            idx = number_emoji_map[message]
            if idx < len(options):
                return {'success': True, 'value': options[idx]}
        
        # Tenta número normal
        if message.isdigit():
            idx = int(message) - 1
            if 0 <= idx < len(options):
                return {'success': True, 'value': options[idx]}
        
        # Tenta match exato ou parcial
        normalized_msg = cls.normalize(message)
        for i, option in enumerate(options):
            if cls.normalize(option) == normalized_msg:
                return {'success': True, 'value': option}
            if cls.normalize(option) in normalized_msg or normalized_msg in cls.normalize(option):
                return {'success': True, 'value': option}
        
        return {'success': False}
    
    @classmethod
    def parse_likert(cls, message: str) -> Dict[str, Any]:
        """Parse de escala Likert"""
        message = message.strip()
        
        # Mapeia emojis e números para valores
        emoji_map = {
            '😞': 1, '🙁': 2, '😐': 3, '🙂': 4, '😄': 5,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '1️⃣': 1, '2️⃣': 2, '3️⃣': 3, '4️⃣': 4, '5️⃣': 5
        }
        
        # Verifica número direto
        if message in emoji_map:
            return {'success': True, 'value': emoji_map[message]}
        
        # Verifica palavras-chave
        normalized = cls.normalize(message)
        keywords = {
            1: ['discordo totalmente', 'discordo muito', 'pessimo', 'horrivel'],
            2: ['discordo', 'ruim', 'mal'],
            3: ['neutro', 'medio', 'mais ou menos', 'talvez'],
            4: ['concordo', 'bom', 'bem'],
            5: ['concordo totalmente', 'concordo muito', 'otimo', 'excelente']
        }
        
        for value, words in keywords.items():
            for word in words:
                if cls.normalize(word) in normalized:
                    return {'success': True, 'value': value}
        
        return {'success': False}
    
    @classmethod
    def parse_text(cls, message: str) -> Dict[str, Any]:
        """Parse de texto livre"""
        message = message.strip()
        if len(message) > 0:
            return {'success': True, 'value': message[:500]}  # Limita tamanho
        return {'success': False}
    
    @classmethod
    def llm_parse(cls, message: str, question: dict, attempt_context: dict = None) -> Dict[str, Any]:
        """
        Usa LLM para analisar mensagens ambíguas:
        - Identifica se é pergunta sobre o questionário
        - Interpreta respostas ambíguas
        - Detecta tentativas de desvio
        - Detecta intenção de pular pergunta
        """
        try:
            qtype = question.get('type', 'text')
            required = question.get('required', False)
            question_id = question.get('id', 0)
            attempt_context = attempt_context or {}
            clarification_count = attempt_context.get('clarification_count', 0)
            skipped_count = attempt_context.get('skipped_questions', 0)
            
            # Determina quais perguntas são obrigatórias baseado no JSON
            # IDs das perguntas obrigatórias: 4 (unidade), 5 (área/setor), 7 (tipo contratação)
            required_question_ids = [4, 5, 7]
            is_required = question_id in required_question_ids or required
            
            # Primeiro, analisa se é pergunta ou resposta
            analysis_prompt = f"""Você está auxiliando no questionário psicossocial da Vocal Silence.

SOBRE A VOCAL SILENCE:
A Vocal Silence tem a missão de tornar o cuidado com a saúde mental um direito acessível, utilizando inteligência artificial para fortalecer a autonomia e o autoconhecimento individual e coletivo. 
"Vocal" representa o impulso de falar, de ser ouvido, de comunicar dores, emoções e necessidades.
"Silence" é a pausa necessária, o espaço de escuta, de reflexão, de reconexão com o que sentimos mas ainda não sabemos nomear.
Cuidar da saúde mental não é apenas permitir que as pessoas falem — é garantir que sejam compreendidas.
Para saber mais: https://www.vocalsilence.com/behind-the-listening

SOBRE O QUESTIONÁRIO:
- Total de 41 perguntas para análise psicossocial
- Objetivo: Melhorar a saúde mental dos colaboradores da empresa
- Respostas são anônimas e confidenciais
- Pode haver perguntas adicionais se identificarmos algum risco
- Perguntas opcionais podem ser puladas (máximo 5)
- Perguntas obrigatórias: unidade de trabalho (ID 4), área/setor (ID 5) e tipo de contratação (ID 7)

CONTEXTO DA PERGUNTA ATUAL:
Pergunta ID {question_id}: "{question.get('question', '')}"
Tipo: {qtype}
{f"Opções: {question.get('options', [])}" if question.get('options') else ""}
Esta pergunta é: {"OBRIGATÓRIA - não pode ser pulada" if is_required else "OPCIONAL - pode ser pulada"}
Perguntas já puladas: {skipped_count}/5
Esclarecimentos já fornecidos nesta pergunta: {clarification_count}

Mensagem do usuário: "{message}"

ANÁLISE:
Determine a intenção do usuário:
1. "question" - Fazendo pergunta sobre o questionário ou pedindo esclarecimento
2. "answer" - Tentando responder a pergunta atual
3. "skip_request" - Querendo pular/não responder a pergunta
4. "off_topic" - Assunto não relacionado ao questionário

REGRAS PARA DETECTAR INTENÇÃO DE PULAR:
- Frases explícitas: "pular", "próxima", "passar", "pulo", "skip" = skip_request
- Recusa em responder: "não quero responder", "prefiro não responder", "não vou responder" = skip_request
- Incerteza genuína: "não sei", "não tenho certeza", "não faço ideia" = skip_request
- Marcadores vazios: "-", "...", "n/a", "NA", "não aplicável" = skip_request
- IMPORTANTE: "não sei" sobre um TERMO (ex: "não sei o que é CLT") = question, não skip_request
- Se a pergunta é OBRIGATÓRIA, ainda detecte skip_request mas será tratado diferente

REGRAS PARA PERGUNTAS/ESCLARECIMENTOS:
- Perguntas sobre termos: "o que é CLT/PJ/turno/assédio?" = question
- Perguntas sobre o processo: "quantas perguntas faltam?", "posso pular?" = question
- Perguntas sobre o questionário: "quem está conduzindo?", "quem aplica?", "quem faz o questionário?" = question
- Pedidos de esclarecimento: "não entendi", "como assim?", "pode explicar?" = question
- Use tom empático e acolhedor nas respostas de esclarecimento

REGRAS ESPECIAIS PARA PERGUNTAS SOBRE O PROCESSO:
- "Quem está conduzindo o questionário?" = question (resposta: Vocal Silence com IA)
- "Quem aplica este questionário?" = question
- "Quem está fazendo as perguntas?" = question
- NUNCA interprete essas perguntas como respostas!

EXEMPLOS DE CLASSIFICAÇÃO:
- "pular" = skip_request
- "não quero responder isso" = skip_request
- "não sei" (sem contexto adicional) = skip_request
- "não sei o que é PJ" = question (pergunta sobre termo)
- "posso pular esta pergunta?" = question (com intenção secundária de pular)
- "Quem está conduzindo o questionário?" = question (NUNCA answer)
- "CLT" = answer
- "acho que é CLT" = answer
- "qual o clima hoje?" = off_topic

Responda APENAS em JSON:
{{
  "intent": "question" | "answer" | "skip_request" | "off_topic",
  "confidence": 0.0-1.0,
  "wants_to_skip": true/false,
  "clarification_response": "resposta empática e clara se for pergunta válida (máximo 3 linhas)",
  "interpreted_value": "valor interpretado se for resposta",
  "should_insist": true/false (true se já forneceu 2+ esclarecimentos),
  "reasoning": "explicação breve da decisão"
}}"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": analysis_prompt}],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Log para debug
            print(f"[LLM Parse] Intent: {analysis.get('intent')}, Confidence: {analysis.get('confidence')}, Message: {message[:50]}")
            
            # Se detectou intenção de pular
            if analysis.get('intent') == 'skip_request' or analysis.get('wants_to_skip'):
                return {
                    'success': False,
                    'wants_to_skip': True,
                    'confidence': analysis.get('confidence', 0.8),
                    'llm_used': True,
                    'reasoning': analysis.get('reasoning', 'Usuário quer pular a pergunta')
                }
            
            # Se for pergunta válida sobre o questionário
            if analysis.get('intent') == 'question' and analysis.get('confidence', 0) > 0.6:
                # Se já deu muitos esclarecimentos, insiste na resposta
                if analysis.get('should_insist') or clarification_count >= 2:
                    return {
                        'success': False,
                        'is_clarification': True,
                        'clarification_limit_reached': True,
                        'message': "Já forneci esclarecimentos sobre esta pergunta. Por favor, escolha uma das opções apresentadas.",
                        'llm_used': True
                    }
                
                return {
                    'success': False,
                    'is_clarification': True,
                    'clarification_response': analysis.get('clarification_response', 'Vou esclarecer sua dúvida.'),
                    'llm_used': True
                }
            
            # Se for off-topic
            if analysis.get('intent') == 'off_topic' and analysis.get('confidence', 0) > 0.7:
                return {
                    'success': False,
                    'is_off_topic': True,
                    'message': "Por favor, vamos focar no questionário de saúde ocupacional. Responda a pergunta apresentada.",
                    'llm_used': True
                }
            
            # Se chegou aqui, tenta interpretar como resposta
            if qtype == 'likert':
                interpret_prompt = f"""O usuário respondeu: "{message}"
Para uma pergunta Likert (escala 1-5) sobre: "{question.get('question', '')}"

Interprete a resposta considerando:
1 = Discordo totalmente
2 = Discordo  
3 = Neutro
4 = Concordo
5 = Concordo totalmente

Exemplos de interpretação:
- "mais ou menos" = 3
- "sim" ou "concordo" = 4
- "com certeza" ou "totalmente" = 5
- "não" ou "discordo" = 2
- "de jeito nenhum" = 1

Responda APENAS em formato JSON.

Se possível interpretar com alta confiança, retorne em formato JSON:
{{"value": 1-5, "confidence": 0.0-1.0}}
Se não for possível interpretar claramente, retorne em JSON:
{{"value": null, "confidence": 0}}"""
            
            elif qtype == 'multiple choice':
                options = question.get('options', [])
                interpret_prompt = f"""O usuário respondeu: "{message}"
Para a pergunta: "{question.get('question', '')}"
Opções disponíveis: {options}

Tente identificar qual opção o usuário escolheu.
Considere abreviações, sinônimos e respostas parciais.

Exemplos:
- Se opções são ["CLT", "PJ", "Estagiário"] e usuário disse "sou CLT", interprete como "CLT"
- Se usuário disse apenas parte da opção, mas é identificável, aceite

Se possível interpretar com alta confiança, retorne em formato JSON:
{{"value": "opção exata da lista", "confidence": 0.0-1.0}}
Se não for possível interpretar claramente, retorne em JSON:
{{"value": null, "confidence": 0}}"""
            
            else:  # text
                # Para texto livre, aceita qualquer resposta não vazia
                if len(message.strip()) > 0:
                    return {'success': True, 'value': message[:500], 'llm_used': True}
                return {'success': False, 'llm_used': True}
            
            interpretation = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": interpret_prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(interpretation.choices[0].message.content)
            
            if result.get('value') and result.get('confidence', 0) > 0.7:
                return {
                    'success': True,
                    'value': result['value'],
                    'llm_used': True,
                    'interpretation_confidence': result.get('confidence')
                }
            
            # Se não conseguiu interpretar
            return {
                'success': False,
                'llm_used': True,
                'could_not_interpret': True,
                'message': "Não consegui entender sua resposta. Por favor, escolha uma das opções apresentadas."
            }
            
        except Exception as e:
            print(f"[LLM Parse Error] {e}")
            return {'success': False, 'llm_used': True, 'error': str(e)}

# =========================
# State Machine (modificado para gerenciar crise com LLM)
# =========================
class QuestionnaireStateMachine:
    
    def __init__(self, sender_id: str):
        self.sender_id = sender_id
        self.state = None
        self.current_question_index = 0
        self.phase1_data = []
        self.followup_data = {"aprofundamento": [], "origem_riscos": {}}
        self.trigger_dimensions = []
        self.attempt_counts = {}
        self.skipped_questions = 0
        self.crisis_manager = None  # Gerenciador de crise
        self.pre_crisis_state = None  # Estado antes da crise
        self.pre_crisis_question_index = None  # Índice da pergunta antes da crise
        self.load_questionnaire()
        self.load_state()
    
    def load_questionnaire(self):
        """Carrega questionário do arquivo JSON"""
        global _questionnaire_cache
        if _questionnaire_cache is None:
            with open("questionario.json", "r", encoding="utf-8") as f:
                _questionnaire_cache = json.load(f)
        self.questionnaire = _questionnaire_cache.get("questionnaire", [])
    
    def load_state(self):
        """Carrega estado do banco de dados"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT current_state, current_question_index, phase1_data, 
                           followup_data, trigger_dimensions, attempt_counts, skipped_questions,
                           pre_crisis_state, pre_crisis_question_index
                    FROM questionnaire_state
                    WHERE sender_id = %s
                """, (self.sender_id,))
                
                result = cur.fetchone()
                if result:
                    self.state = State(result[0])
                    self.current_question_index = result[1]
                    self.phase1_data = json.loads(result[2]) if result[2] else []
                    self.followup_data = json.loads(result[3]) if result[3] else {"aprofundamento": [], "origem_riscos": {}}
                    self.trigger_dimensions = json.loads(result[4]) if result[4] else []
                    self.attempt_counts = json.loads(result[5]) if result[5] else {}
                    self.skipped_questions = result[6] or 0
                    self.pre_crisis_state = State(result[7]) if result[7] else None
                    self.pre_crisis_question_index = result[8]
                    
                    # Se estava em emergência, carrega o gerenciador de crise
                    if self.state == State.EMERGENCY:
                        # Carrega gerenciador com estado existente
                        self.crisis_manager = CrisisManager(self.sender_id, load_existing=True)
                        # Garante que tem um tipo de crise válido
                        if not self.crisis_manager.crisis_type:
                            self.crisis_manager.crisis_type = 'unknown'
                            print(f"[State Machine] AVISO: crisis_type estava vazio, definindo como 'unknown'")
                        print(f"[State Machine] Carregado gerenciador de crise: tipo={self.crisis_manager.crisis_type}")
                else:
                    self.state = State.WELCOME
                    self.save_state()
    
    def save_state(self):
        """Salva estado no banco de dados"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO questionnaire_state 
                        (sender_id, current_state, current_question_index, phase1_data, 
                         followup_data, trigger_dimensions, attempt_counts, skipped_questions,
                         pre_crisis_state, pre_crisis_question_index, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (sender_id) DO UPDATE SET
                            current_state = EXCLUDED.current_state,
                            current_question_index = EXCLUDED.current_question_index,
                            phase1_data = EXCLUDED.phase1_data,
                            followup_data = EXCLUDED.followup_data,
                            trigger_dimensions = EXCLUDED.trigger_dimensions,
                            attempt_counts = EXCLUDED.attempt_counts,
                            skipped_questions = EXCLUDED.skipped_questions,
                            pre_crisis_state = EXCLUDED.pre_crisis_state,
                            pre_crisis_question_index = EXCLUDED.pre_crisis_question_index,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        self.sender_id,
                        self.state.value,
                        self.current_question_index,
                        json.dumps(self.phase1_data, ensure_ascii=False),
                        json.dumps(self.followup_data, ensure_ascii=False),
                        json.dumps(self.trigger_dimensions, ensure_ascii=False),
                        json.dumps(self.attempt_counts),
                        self.skipped_questions,
                        self.pre_crisis_state.value if self.pre_crisis_state else None,
                        self.pre_crisis_question_index,
                        datetime.utcnow()
                    ))
        except Exception as e:
            print(f"[DB Save State Error] {e}")
            print("[WARNING] Não foi possível salvar o estado no banco de dados")
    
    def log_interaction(self, message_received: str, message_sent: str, 
                       llm_used: bool = False, safety_triggered: bool = False, 
                       safety_metadata: dict = None, metadata: dict = None):
        """Registra interação no log com campos de segurança aprimorados"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                state_before = self.state.value if self.state else None
                
                # Combina metadados
                full_metadata = metadata or {}
                if safety_metadata:
                    full_metadata['safety'] = safety_metadata
                
                cur.execute("""
                    INSERT INTO questionnaire_logs
                    (sender_id, state_before, state_after, message_received, message_sent, 
                     llm_used, safety_triggered, safety_screening_model, 
                     safety_screening_confidence, safety_detailed_check, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.sender_id,
                    state_before,
                    self.state.value if self.state else None,
                    message_received[:1000],
                    message_sent[:1000],
                    llm_used,
                    safety_triggered,
                    safety_metadata.get('screening_model') if safety_metadata else None,
                    safety_metadata.get('confidence') if safety_metadata else None,
                    safety_metadata.get('detailed_check', False) if safety_metadata else False,
                    json.dumps(full_metadata) if full_metadata else None
                ))
    
    def enter_crisis_mode(self, crisis_type: str, initial_safety_score: int = 3):
        """Entra em modo de crise, salvando estado atual"""
        # Salva estado antes da crise
        self.pre_crisis_state = self.state
        self.pre_crisis_question_index = self.current_question_index
        
        # Muda para estado de emergência
        self.state = State.EMERGENCY
        
        # Cria gerenciador de crise SEM carregar estado existente (é uma nova crise)
        self.crisis_manager = CrisisManager(self.sender_id, load_existing=False)
        self.crisis_manager.crisis_type = crisis_type
        self.crisis_manager.safety_score = initial_safety_score
        # Salva o estado inicial da crise
        self.crisis_manager.save_crisis_state()
        
        self.save_state()
    
    def exit_crisis_mode(self):
        """Sai do modo de crise e retorna ao questionário"""
        if self.pre_crisis_state:
            # Restaura estado anterior
            self.state = self.pre_crisis_state
            self.current_question_index = self.pre_crisis_question_index
            self.pre_crisis_state = None
            self.pre_crisis_question_index = None
        else:
            # Se não tinha estado anterior, vai para welcome
            self.state = State.WELCOME
            self.current_question_index = 0
        
        self.crisis_manager = None
        self.save_state()
    
    def reset(self):
        """Reinicia questionário"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Remove estado do questionário
                cur.execute("DELETE FROM questionnaire_state WHERE sender_id = %s", (self.sender_id,))
                # Remove estado de crise se existir
                cur.execute("UPDATE crisis_state SET active = false, resolution_reason = 'reset_questionnaire', resolved_at = %s WHERE sender_id = %s AND active = true", (datetime.utcnow(), self.sender_id))
        
        self.state = State.WELCOME
        self.current_question_index = 0
        self.phase1_data = []
        self.followup_data = {"aprofundamento": [], "origem_riscos": {}}
        self.trigger_dimensions = []
        self.attempt_counts = {}
        self.skipped_questions = 0
        self.pre_crisis_state = None
        self.pre_crisis_question_index = None
        self.crisis_manager = None
        self.save_state()
    
    def format_question(self, question: dict, position: int = None, total: int = None) -> str:
        """Formata pergunta para WhatsApp - versão otimizada"""
        qtype = question.get('type', 'text')
        text = question.get('question', '')
        
        # Barra de progresso visual
        progress_bar = ""
        if position and total:
            percentage = int((position / total) * 100)
            filled = int(percentage / 10)  # Divide por 10 para ter 10 blocos
            empty = 10 - filled
            progress_bar = f"{'▓' * filled}{'░' * empty} {percentage}%\n"
            
            # Adiciona indicador de seção
            if position == 1:
                progress_bar = f"🚀 *Iniciando questionário*\n{progress_bar}"
            elif position == total:
                progress_bar = f"🏁 *Última pergunta!*\n{progress_bar}"
            elif position == total // 2:
                progress_bar = f"⭐ *Metade do caminho!*\n{progress_bar}"
            
            text = f"*Pergunta {position} de {total}*\n{progress_bar}\n*{text}*"
        else:
            text = f"*{text}*"
        
        # Adiciona formatação baseada no tipo (sem linhas divisórias)
        if qtype == 'likert':
            text += "\n\n"
            text += "1️⃣ 😞 Discordo totalmente\n"
            text += "2️⃣ 🙁 Discordo\n"
            text += "3️⃣ 😐 Neutro\n"
            text += "4️⃣ 🙂 Concordo\n"
            text += "5️⃣ 😄 Concordo totalmente"
            text += "\n\n💡 _Responda com número, texto ou áudio_ 🎤"
        elif qtype == 'multiple choice':
            text += "\n"
            options = question.get('options', [])
            # Usa emojis de números (agora suportados no parser!)
            number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣']
            for i, opt in enumerate(options):
                if i < len(number_emojis):
                    text += f"\n{number_emojis[i]} {opt}"
                else:
                    text += f"\n{i+1}) {opt}"
            text += "\n\n💡 _Responda com número, texto ou áudio_ 🎤"
        elif qtype == 'text':
            text += "\n\n✍️ _Digite sua resposta livremente_"
            text += "\n\n💡 _Responda com texto ou áudio_ 🎤"
        
        return text
    
    def format_followup_question(self, question: dict, position: int, total: int) -> str:
        """Formata pergunta de follow-up - versão otimizada"""
        # Barra de progresso para follow-up
        percentage = int((position / total) * 100)
        filled = int(percentage / 10)
        empty = 10 - filled
        progress_bar = f"{'▓' * filled}{'░' * empty} {percentage}%"
        
        text = f"🔍 *Aprofundamento {position}/{total}*\n"
        text += f"{progress_bar}\n\n"
        text += f"*{question['question']}*\n"
        
        options = question.get('options', [])
        number_emojis = ['1️⃣', '2️⃣', '3️⃣']
        for i, opt in enumerate(options):
            if i < len(number_emojis):
                text += f"\n{number_emojis[i]} {opt}"
            else:
                text += f"\n{i+1}) {opt}"
        
        text += "\n\n💡 _Responda com número, texto ou áudio_ 🎤"
        return text
    
    def format_origin_question(self, question: dict, position: int, total: int, dimension_desc: str = None) -> str:
        """Formata pergunta de origem dos riscos - versão otimizada"""
        # Barra de progresso
        percentage = int((position / total) * 100)
        filled = int(percentage / 10)
        empty = 10 - filled
        progress_bar = f"{'▓' * filled}{'░' * empty} {percentage}%"
        
        text = f"🔍 *Origem dos Riscos {position}/{total}*\n"
        text += f"{progress_bar}\n\n"
        
        # Se tem descrição da dimensão, adiciona
        if dimension_desc:
            text += f"⚠️ _Riscos identificados {dimension_desc}_\n\n"
        
        text += f"*{question['question']}*"
        
        # Formata baseado no tipo
        if question.get('type') == 'multiple choice':
            text += "\n"
            options = question.get('options', [])
            number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣']
            for i, opt in enumerate(options):
                if i < len(number_emojis):
                    text += f"\n{number_emojis[i]} {opt}"
                else:
                    text += f"\n{i+1}) {opt}"
            text += "\n\n💡 _Responda com número, texto ou áudio_ 🎤"
        else:
            text += "\n\n✍️ _Digite sua resposta livremente_"
            text += "\n\n💡 _Responda com texto ou áudio_ 🎤"
        
        return text
    
    def get_welcome_message(self) -> str:
        """Mensagem de boas-vindas"""
        return """👋 Olá! Somos da Vocal Silence e queremos ouvir como você se sente no ambiente de trabalho.

Este é um questionário psicossocial que vai mapear fatores que impactam seu dia a dia e, com isso, oferecer subsídios para que sua empresa construa planos de ação orientados por dados.

🔒 Suas respostas são anônimas e tratadas com sigilo.
⏱️ Em poucos minutos você contribui para mudanças reais.
📌 A participação é voluntária – você pode parar a qualquer momento.
⚕️ Importante: este questionário não é avaliação médica e não fornece diagnóstico.

👉 Podemos começar?"""
    
    def handle_consent(self, message: str) -> str:
        """Processa consentimento"""
        normalized = ResponseParser.normalize(message)
        
        if any(word in normalized for word in ['sim', 'yes', 'ok', 'vamos', 'pode', 'aceito', 'concordo']):
            self.state = State.PHASE1_QUESTIONS
            self.save_state()
            
            # Explica Likert antes da primeira pergunta
            intro = "Ótimo! Vamos começar.\n\n"
            intro += "📊 Algumas perguntas usam uma escala de 1 a 5:\n"
            intro += "• 1 = Discordo totalmente\n• 5 = Concordo totalmente\n"
            intro += "Você pode responder com número, emoji ou texto.\n\n"
            
            # Primeira pergunta
            question = self.questionnaire[0]
            self.phase1_data = [question.copy()]
            intro += self.format_question(question, 1, len(self.questionnaire))
            
            return intro
        
        elif any(word in normalized for word in ['nao', 'não', 'no', 'depois', 'pare']):
            self.reset()
            return "Sem problemas! Quando quiser participar, é só enviar uma mensagem. Até logo!"
        
        else:
            return "Por favor, responda 'sim' para começar ou 'não' para cancelar."
    
    def handle_phase1_question(self, message: str) -> Tuple[str, bool]:
        """Processa resposta da Fase 1 com análise inteligente"""
        if self.current_question_index >= len(self.questionnaire):
            self.state = State.ASSESSMENT
            self.save_state()
            return self.do_assessment()
        
        question = self.phase1_data[self.current_question_index]
        qtype = question.get('type', 'text')
        required = question.get('required', False)
        
        # Identifica ID da pergunta para contar tentativas
        q_id = f"q_{question.get('id', self.current_question_index)}"
        self.attempt_counts[q_id] = self.attempt_counts.get(q_id, 0) + 1
        attempts = self.attempt_counts[q_id]
        
        # Rastreia esclarecimentos separadamente
        clarification_key = f"{q_id}_clarifications"
        clarification_count = self.attempt_counts.get(clarification_key, 0)
        
        # Parse da resposta
        parsed = None
        llm_used = False
        
        if qtype == 'multiple choice':
            parsed = ResponseParser.parse_multiple_choice(message, question.get('options', []))
        elif qtype == 'likert':
            parsed = ResponseParser.parse_likert(message)
        else:
            parsed = ResponseParser.parse_text(message)
        
        # Se falhou o parse rápido, usa LLM
        if not parsed.get('success'):
            attempt_context = {
                'clarification_count': clarification_count,
                'skipped_questions': self.skipped_questions
            }
            parsed = ResponseParser.llm_parse(message, question, attempt_context)
            llm_used = parsed.get('llm_used', False)
            
            # Se foi identificado como tentativa de pular
            if parsed.get('wants_to_skip'):
                if required:
                    # Pergunta obrigatória - não pode pular
                    return (f"⚠️ Esta pergunta é obrigatória e não pode ser pulada.\n\n" +
                           f"Por favor, responda para continuar:\n" +
                           self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                           llm_used)
                else:
                    # Pergunta opcional - pode pular mas com limite
                    if self.skipped_questions >= 5:
                        # Já atingiu o limite
                        self.reset()
                        return ("❌ Você já pulou o máximo de 5 perguntas permitidas. " +
                               "O questionário será reiniciado para garantir dados consistentes. " +
                               "Digite qualquer mensagem para começar novamente.", llm_used)
                    else:
                        # Ainda pode pular
                        remaining_skips = 5 - self.skipped_questions - 1
                        question['response'] = None
                        question['desconsiderada'] = True
                        self.skipped_questions += 1
                        self.current_question_index += 1
                        
                        # Adiciona próxima pergunta se existir
                        if self.current_question_index < len(self.questionnaire):
                            if self.current_question_index >= len(self.phase1_data):
                                next_q = self.questionnaire[self.current_question_index].copy()
                                self.phase1_data.append(next_q)
                        
                        self.save_state()
                        
                        # Verifica se terminou o questionário
                        if self.current_question_index >= len(self.questionnaire):
                            self.state = State.ASSESSMENT
                            self.save_state()
                            return (self.do_assessment(), llm_used)
                        
                        # Mensagem informativa sobre o limite
                        skip_info = ""
                        if remaining_skips > 0:
                            skip_info = f"\n💡 Você ainda pode pular {remaining_skips} pergunta{'s' if remaining_skips > 1 else ''}."
                        else:
                            skip_info = "\n⚠️ Atenção: Você atingiu o limite de perguntas que podem ser puladas. Se ultrapassar o limite, o questionário será reiniciado."
                        
                        return (f"✔ Pergunta pulada.{skip_info}\n\n" +
                               self.format_question(self.questionnaire[self.current_question_index], 
                                                  self.current_question_index + 1, 
                                                  len(self.questionnaire)),
                               llm_used)
            
            # Se foi identificado como pergunta/esclarecimento
            if parsed.get('is_clarification'):
                self.attempt_counts[clarification_key] = clarification_count + 1
                
                if parsed.get('clarification_limit_reached'):
                    # Insiste na resposta
                    return (f"⚠️ {parsed.get('message')}\n\n" + 
                           self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                           llm_used)
                
                # Fornece esclarecimento e reapresenta a pergunta
                clarification = parsed.get('clarification_response', 'Vou esclarecer sua dúvida.')
                return (f"💬 {clarification}\n\n📝 Agora, por favor, responda:\n" +
                       self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                       llm_used)
            
            # Se foi identificado como off-topic
            if parsed.get('is_off_topic'):
                return (f"⚠️ {parsed.get('message')}\n\n" +
                       self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                       llm_used)
            
            # Se não conseguiu interpretar
            if parsed.get('could_not_interpret'):
                # Conta como tentativa falha
                if required:
                    if attempts >= 5:
                        self.reset()
                        return ("Notamos que algumas respostas parecem inconsistentes. Este questionário ajuda a empresa a compreender melhor o ambiente de trabalho. Vamos reiniciá-lo, pois não foi preenchido corretamente. Digite qualquer mensagem para começar novamente.", llm_used)
                    else:
                        return (f"❌ {parsed.get('message', 'Não consegui entender sua resposta.')}\n\n" +
                               f"(Tentativa {attempts}/5)\n" +
                               self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                               llm_used)
                else:
                    if attempts >= 3:
                        # Pula pergunta após 3 tentativas
                        question['response'] = None
                        question['desconsiderada'] = True
                        self.skipped_questions += 1
                        
                        if self.skipped_questions >= 5:
                            self.reset()
                            return ("Notamos que muitas perguntas foram puladas e, por isso, não é possível continuar. Este questionário ajuda a empresa a compreender melhor o ambiente de trabalho. Vamos reiniciá-lo para que possa ser preenchido corretamente. Digite qualquer mensagem para começar novamente.", llm_used)
                        
                        self.current_question_index += 1
                        
                        if self.current_question_index < len(self.questionnaire):
                            if self.current_question_index >= len(self.phase1_data):
                                next_q = self.questionnaire[self.current_question_index].copy()
                                self.phase1_data.append(next_q)
                        
                        self.save_state()
                        
                        if self.current_question_index >= len(self.questionnaire):
                            self.state = State.ASSESSMENT
                            self.save_state()
                            return (self.do_assessment(), llm_used)
                        
                        return (f"Vamos pular esta pergunta.\n\n" +
                               self.format_question(self.questionnaire[self.current_question_index], 
                                                  self.current_question_index + 1, 
                                                  len(self.questionnaire)),
                               llm_used)
                    else:
                        return (f"❌ Não consegui entender. Tente responder de forma mais clara.\n\n" +
                               f"(Tentativa {attempts}/3)\n" +
                               self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                               llm_used)
        
        # Processa resultado bem-sucedido
        if parsed.get('success'):
            # Validação adicional para multiple choice
            if qtype == 'multiple choice':
                valid_options = question.get('options', [])
                if parsed['value'] not in valid_options:
                    # Resposta inválida - não está nas opções
                    return (f"❌ Resposta inválida. Por favor, escolha uma das opções:\n" +
                           self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                           llm_used)
            elif qtype == 'likert':
                # Valida se é um valor válido de Likert (1-5)
                try:
                    likert_value = int(parsed['value'])
                    if likert_value < 1 or likert_value > 5:
                        return (f"❌ Resposta inválida. Por favor, escolha um valor de 1 a 5:\n" +
                               self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                               llm_used)
                except (ValueError, TypeError):
                    return (f"❌ Resposta inválida. Por favor, escolha um valor de 1 a 5:\n" +
                           self.format_question(question, self.current_question_index + 1, len(self.questionnaire)),
                           llm_used)
            
            # Resposta válida - reseta contadores
            question['response'] = parsed['value']
            question['desconsiderada'] = False
            self.attempt_counts[q_id] = 0  # Reset tentativas
            self.attempt_counts[clarification_key] = 0  # Reset esclarecimentos
            
            # Próxima pergunta
            self.current_question_index += 1
            
            if self.current_question_index >= len(self.questionnaire):
                self.state = State.ASSESSMENT
                self.save_state()
                return (self.do_assessment(), llm_used)
            else:
                # Adiciona próxima pergunta aos dados
                if self.current_question_index < len(self.questionnaire):
                    next_q = self.questionnaire[self.current_question_index].copy()
                    self.phase1_data.append(next_q)
                
                self.save_state()
                
                # Confirmação breve + próxima pergunta
                confidence_indicator = ""
                if parsed.get('interpretation_confidence'):
                    conf = parsed['interpretation_confidence']
                    if conf < 0.85:
                        confidence_indicator = " (interpretado)"
                
                confirmation = f"✔ Resposta registrada{confidence_indicator}."
                next_question = self.format_question(
                    self.questionnaire[self.current_question_index],
                    self.current_question_index + 1,
                    len(self.questionnaire)
                )
                
                return (f"{confirmation}\n\n{next_question}", llm_used)
        
        # Fallback - não deveria chegar aqui
        return ("Houve um erro ao processar sua resposta. Por favor, tente novamente.", llm_used)


    def do_assessment(self) -> str:
        """Realiza avaliação e determina próximos passos"""
        # Calcula médias por dimensão
        dimension_scores = {}
        target_dimensions = {
            "Qualidade do sono e disposição",
            "Ânimo e motivação", 
            "Estresse e ansiedade",
            "Equilíbrio vida-trabalho",
            "Exigências de tempo no trabalho"
        }
        
        for item in self.phase1_data:
            if item.get('type', '').lower() == 'likert' and not item.get('desconsiderada'):
                if item.get('response') is not None:
                    dim = item.get('dimension', '')
                    if dim not in dimension_scores:
                        dimension_scores[dim] = []
                    try:
                        dimension_scores[dim].append(float(item['response']))
                    except:
                        pass
        
        # Identifica dimensões com risco
        self.trigger_dimensions = []
        for dim, scores in dimension_scores.items():
            if scores and ResponseParser.normalize(dim) in [ResponseParser.normalize(d) for d in target_dimensions]:
                avg = sum(scores) / len(scores)
                if avg <= 3.0:  # ALTO ou MODERADO
                    self.trigger_dimensions.append(dim)
        
        # Salva fase 1
        self._save_to_s3('phase1')
        
        if self.trigger_dimensions:
            self.state = State.FOLLOWUP_QUESTIONS  # Vai direto para FOLLOWUP_QUESTIONS
            self.current_question_index = 0  # Inicializa o índice
            self.save_state()
            # Retorna lista com duas mensagens: introdução + primeira pergunta
            q = FOLLOWUP_QUESTIONS[0]
            return [
                "Percebemos alguns sinais de risco nesta etapa. "
                "Vamos fazer algumas perguntas de aprofundamento para entender melhor o que está acontecendo.",
                self.format_followup_question(q, 1, 6)
            ]
        else:
            # Sem riscos, finaliza
            self.state = State.COMPLETION
            self.save_state()
            return self.get_completion_message()
    
    def handle_followup_questions(self, message: str) -> Tuple[str, bool]:
        """Processa perguntas de follow-up com análise inteligente"""
        llm_used = False
        
        if self.current_question_index < len(FOLLOWUP_QUESTIONS):
            question = FOLLOWUP_QUESTIONS[self.current_question_index]
            
            # Rastreia esclarecimentos
            q_id = f"followup_{question['id']}"
            clarification_key = f"{q_id}_clarifications"
            clarification_count = self.attempt_counts.get(clarification_key, 0)
            
            parsed = ResponseParser.parse_multiple_choice(message, question['options'])
            
            if not parsed.get('success'):
                # Cria estrutura de pergunta compatível
                question_dict = {
                    'type': 'multiple choice',
                    'options': question['options'],
                    'question': question['question']
                }
                attempt_context = {'clarification_count': clarification_count}
                parsed = ResponseParser.llm_parse(message, question_dict, attempt_context)
                llm_used = True
                
                # Processa esclarecimentos
                if parsed.get('is_clarification'):
                    self.attempt_counts[clarification_key] = clarification_count + 1
                    
                    if parsed.get('clarification_limit_reached'):
                        return (f"⚠️ {parsed.get('message')}\n\n" +
                               self.format_followup_question(question, self.current_question_index + 1, 6),
                               llm_used)
                    
                    clarification = parsed.get('clarification_response', '')
                    return (f"💬 {clarification}\n\n📝 Por favor, responda:\n" +
                           self.format_followup_question(question, self.current_question_index + 1, 6),
                           llm_used)
                
                if parsed.get('is_off_topic'):
                    return (f"⚠️ {parsed.get('message')}\n\n" +
                           self.format_followup_question(question, self.current_question_index + 1, 6),
                           llm_used)
            
            if parsed.get('success'):
                # Validação adicional: verifica se o valor está realmente nas opções
                if parsed['value'] not in question['options']:
                    # Resposta inválida
                    return (f"❌ Resposta inválida. Por favor, escolha uma das opções:\n" +
                           self.format_followup_question(question, self.current_question_index + 1, 6),
                           llm_used)
                
                # Salva resposta
                q_copy = question.copy()
                q_copy['response'] = parsed['value']
                self.followup_data['aprofundamento'].append(q_copy)
                
                # Reseta contadores
                self.attempt_counts[clarification_key] = 0
                
                self.current_question_index += 1
                self.save_state()
                
                if self.current_question_index >= len(FOLLOWUP_QUESTIONS):
                    # Passa diretamente para ORIGIN_QUESTIONS
                    self.state = State.ORIGIN_QUESTIONS  # Vai direto para ORIGIN_QUESTIONS
                    self.current_question_index = 0
                    self.save_state()
                    # Retorna lista com duas mensagens: introdução + primeira pergunta de origem
                    dim = self.trigger_dimensions[0] if self.trigger_dimensions else "Risco identificado"
                    dim_description = DIMENSION_DESCRIPTIONS.get(dim, dim)
                    q = ORIGIN_QUESTIONS[0]
                    question_text = f"🔍 *Foram encontrados riscos {dim_description}*\n\n"
                    question_text += f"*1/{len(self.trigger_dimensions)*2} – {q['question']}*"
                    for i, opt in enumerate(q['options'], 1):
                        question_text += f"\n{i}) {opt}"
                    return ([
                        "Vamos fazer algumas outras perguntas para entender melhor a origem destes riscos.",
                        question_text
                    ], llm_used)
                else:
                    next_q = FOLLOWUP_QUESTIONS[self.current_question_index]
                    return (f"✔ Registrado.\n\n{self.format_followup_question(next_q, self.current_question_index + 1, 6)}", llm_used)
            else:
                # Usa as opções da pergunta atual
                if self.current_question_index < len(FOLLOWUP_QUESTIONS):
                    current_q = FOLLOWUP_QUESTIONS[self.current_question_index]
                    options = current_q.get('options', [])
                    if options:
                        options_text = " ou ".join([f"'{opt}'" for opt in options])
                        return (f"Por favor, responda {options_text}.", llm_used)
                return ("Por favor, responda com uma das opções apresentadas.", llm_used)
        
        return ("", llm_used)
    
    def handle_origin_questions(self, message: str) -> Tuple[str, bool]:
        """Processa perguntas de origem com as NOVAS perguntas (2 por dimensão)"""
        # Determina qual dimensão e pergunta atual
        dim_index = self.current_question_index // 2  # Mudado de 3 para 2
        q_index = self.current_question_index % 2    # Mudado de 3 para 2
        
        if dim_index >= len(self.trigger_dimensions):
            # Finaliza
            self._save_to_s3('followups')
            self.state = State.COMPLETION
            self.save_state()
            return (self.get_completion_message(), False)
        
        current_dim = self.trigger_dimensions[dim_index]
        question = ORIGIN_QUESTIONS[q_index]
        
        # Parse da resposta baseado no tipo
        llm_used = False
        if q_index == 0:  # Primeira pergunta (multiple choice)
            parsed = ResponseParser.parse_multiple_choice(message, question['options'])
            
            if not parsed.get('success'):
                # Usa LLM para tentar interpretar
                question_dict = {
                    'type': 'multiple choice',
                    'options': question['options'],
                    'question': question['question']
                }
                parsed = ResponseParser.llm_parse(message, question_dict)
                llm_used = parsed.get('llm_used', False)
                
                if parsed.get('is_clarification'):
                    clarification = parsed.get('clarification_response', '')
                    # Reapresenta a pergunta formatada
                    total_origin = len(self.trigger_dimensions) * 2
                    current_pos = self.current_question_index + 1
                    question_formatted = self.format_origin_question(question, current_pos, total_origin)
                    return (f"💬 {clarification}\n\n📝 Por favor, responda:\n{question_formatted}", llm_used)
                
                if not parsed.get('success'):
                    # Mostra opções com emojis de números
                    return (f"❌ Por favor, escolha uma das opções:\n\n1️⃣ Do ambiente de trabalho\n2️⃣ Da vida pessoal\n3️⃣ Dos dois\n\n💡 _Responda com número, texto ou áudio_ 🎤", llm_used)
            
            response_value = parsed['value']
        else:  # Segunda pergunta (texto livre)
            response_value = message[:500]
        
        # Salva resposta
        if current_dim not in self.followup_data['origem_riscos']:
            self.followup_data['origem_riscos'][current_dim] = []
        
        q_copy = question.copy()
        q_copy['response'] = response_value
        self.followup_data['origem_riscos'][current_dim].append(q_copy)
        
        self.current_question_index += 1
        self.save_state()
        
        # Próxima pergunta
        new_dim_index = self.current_question_index // 2  # Mudado de 3 para 2
        new_q_index = self.current_question_index % 2     # Mudado de 3 para 2
        
        if new_dim_index >= len(self.trigger_dimensions):
            # Finaliza
            self._save_to_s3('followups')
            self.state = State.COMPLETION
            self.save_state()
            return (self.get_completion_message(), llm_used)
        
        new_dim = self.trigger_dimensions[new_dim_index]
        
        # Prepara mensagem para próxima pergunta
        next_question = ORIGIN_QUESTIONS[new_q_index]
        total_origin = len(self.trigger_dimensions) * 2  # Mudado de 3 para 2
        current_pos = self.current_question_index + 1
        
        # Usa o novo método de formatação
        if new_q_index == 0:  # Nova dimensão - mostra descrição
            dim_description = DIMENSION_DESCRIPTIONS.get(new_dim, new_dim)
            question_text = self.format_origin_question(next_question, current_pos, total_origin, dim_description)
        else:
            question_text = self.format_origin_question(next_question, current_pos, total_origin)
        
        return (f"✔ Registrado.\n\n{question_text}", llm_used)
    
    def _save_to_s3(self, phase: str):
        """Salva dados no S3"""
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        
        if phase == 'phase1':
            data = {"questionnaire": self.phase1_data}
            key = f"questionario_state_machine/{self.sender_id}/{timestamp}/phase1.json"
        else:
            data = self.followup_data
            key = f"questionario_state_machine/{self.sender_id}/{timestamp}/followups.json"
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False).encode('utf-8'),
            ContentType="application/json; charset=utf-8"
        )
    
    def get_completion_message(self) -> str:
        """Mensagem de conclusão"""
        self.reset()  # Limpa estado
        return ("✨ Questionário concluído! Agradecemos imensamente sua participação e confiança. "
               "Suas respostas foram registradas com sucesso e serão tratadas com total confidencialidade. "
               "Agora vamos apagar o histórico desta conversa para garantir sua privacidade. Muito obrigado! 🙏")
    
    def get_resume_questionnaire_message(self) -> str:
        """Mensagem ao retomar questionário após crise"""
        current_state_messages = {
            State.PHASE1_QUESTIONS: "Vamos continuar o questionário de onde paramos.",
            State.FOLLOWUP_QUESTIONS: "Vamos continuar com as perguntas de aprofundamento.",
            State.ORIGIN_QUESTIONS: "Vamos continuar explorando as origens dos riscos identificados.",
        }
        
        base_msg = "Que bom que você está melhor! 💚\n\n"
        state_msg = current_state_messages.get(self.state, "Vamos continuar de onde paramos.")
        
        print(f"[Resume] Estado atual: {self.state}, índice: {self.current_question_index}")
        
        # Reapresenta a última pergunta
        if self.state == State.PHASE1_QUESTIONS:
            if self.current_question_index < len(self.questionnaire):
                question = self.questionnaire[self.current_question_index]
                question_text = self.format_question(
                    question,
                    self.current_question_index + 1,
                    len(self.questionnaire)
                )
                print(f"[Resume] Pergunta Phase1: {self.current_question_index + 1}/{len(self.questionnaire)}")
                return f"{base_msg}{state_msg}\n\n{question_text}"
            else:
                print(f"[Resume] Índice Phase1 inválido: {self.current_question_index}/{len(self.questionnaire)}")
        
        elif self.state == State.FOLLOWUP_QUESTIONS:
            if self.current_question_index < len(FOLLOWUP_QUESTIONS):
                question = FOLLOWUP_QUESTIONS[self.current_question_index]
                print(f"[Resume] Pergunta Followup: {self.current_question_index + 1}/6")
                return f"{base_msg}{state_msg}\n\n{self.format_followup_question(question, self.current_question_index + 1, 6)}"
            else:
                print(f"[Resume] Índice Followup inválido: {self.current_question_index}/6")
        
        elif self.state == State.ORIGIN_QUESTIONS:
            dim_index = self.current_question_index // 2  # Mudado de 3 para 2
            q_index = self.current_question_index % 2    # Mudado de 3 para 2
            if dim_index < len(self.trigger_dimensions):
                dim = self.trigger_dimensions[dim_index]
                # Obtém descrição detalhada da dimensão
                dim_description = DIMENSION_DESCRIPTIONS.get(dim, dim)
                question = ORIGIN_QUESTIONS[q_index]
                total = len(self.trigger_dimensions) * 2  # Mudado de 3 para 2
                current_pos = self.current_question_index + 1
                
                if q_index == 0:
                    prefix = f"🔍 *Foram encontrados riscos {dim_description}*\n\n"
                else:
                    prefix = ""
                
                # Formata pergunta baseado no tipo
                if question.get('type') == 'multiple choice':
                    question_text = f"*{current_pos}/{total} – {question['question']}*"
                    for i, opt in enumerate(question['options'], 1):
                        question_text += f"\n{i}) {opt}"
                else:
                    question_text = f"*{current_pos}/{total} – {question['question']}*"
                
                print(f"[Resume] Pergunta Origin: {current_pos}/{total}, dim={dim}")
                return f"{base_msg}{state_msg}\n\n{prefix}{question_text}"
            else:
                print(f"[Resume] Índice Origin inválido: dim_index={dim_index}, trigger_dimensions={len(self.trigger_dimensions)}")
        
        elif self.state == State.CONSENT:
            print(f"[Resume] Retornando para consentimento")
            return f"{base_msg}Vamos retomar onde paramos.\n\n{self.get_welcome_message()}"
            
        else:
            print(f"[Resume] Estado não tratado: {self.state}")
            # Fallback - volta para o início se estado desconhecido
            self.state = State.WELCOME
            self.current_question_index = 0
            self.save_state()
            return f"{base_msg}Vamos reiniciar o questionário.\n\n{self.get_welcome_message()}"
        
        # Garante que sempre retorna algo
        return f"{base_msg}{state_msg}"
    
    def process_message(self, message: str, audio_info: Dict[str, Any] = None) -> str:
        """Processa mensagem e retorna resposta (texto ou áudio transcrito)"""
        llm_used = False
        safety_triggered = False
        safety_metadata = {}
        audio_transcription = None
        
        # Se recebeu áudio, processa transcrição PRIMEIRO
        if audio_info and audio_info.get('media_url'):
            # Determina tipo da pergunta atual para validação de duração
            current_question_type = "text"  # Padrão
            
            if self.state == State.PHASE1_QUESTIONS:
                if self.current_question_index < len(self.questionnaire):
                    current_question_type = self.questionnaire[self.current_question_index].get('type', 'text')
            elif self.state == State.FOLLOWUP_QUESTIONS:
                current_question_type = "multiple choice"  # Follow-up são sempre multiple choice
            elif self.state == State.ORIGIN_QUESTIONS:
                q_index = self.current_question_index % 2
                if q_index == 0:
                    current_question_type = "multiple choice"
                else:
                    current_question_type = "text"
            
            print(f"[Audio] Processando áudio para pergunta tipo: {current_question_type}")
            
            # Processa o áudio
            audio_result = AudioTranscriber.process_audio_message(
                audio_info['media_url'],
                audio_info.get('media_content_type', 'audio/ogg'),
                current_question_type
            )
            
            if not audio_result['success']:
                # Retorna mensagem de erro específica do áudio (duração, etc)
                self.log_interaction(
                    f"[AUDIO: {audio_info['media_url'][:50]}...]",
                    audio_result['message'],
                    False, False, {"audio_error": True, "error": audio_result.get('error')}
                )
                return audio_result['message']
            
            # IMPORTANTE: Substitui a mensagem pela transcrição
            # A partir daqui, tudo funciona como se fosse texto normal
            audio_transcription = audio_result['transcription']
            message = audio_transcription  # Substitui completamente a mensagem
            
            # Adiciona metadados do áudio para o log
            safety_metadata['audio_processed'] = True
            safety_metadata['audio_duration'] = audio_result.get('duration')
            safety_metadata['audio_language'] = audio_result.get('language')
            
            print(f"[Audio] Transcrição substituiu mensagem: {message[:100]}...")
        
        # ==== A PARTIR DAQUI, TUDO FUNCIONA NORMALMENTE ====
        # A mensagem agora é o texto (original ou transcrito do áudio)
        # TODA a lógica abaixo aplica-se igualmente para texto e áudio:
        # - Detecção de comandos especiais (reiniciar, etc)
        # - Modo de emergência/crise
        # - Triagem de segurança (SafetyProtocol)
        # - Possibilidade de pular perguntas
        # - Uso de LLM para interpretar respostas ambíguas
        # - Toda a máquina de estados do questionário
        
        # Verifica comandos especiais primeiro (mesmo durante crise)
        normalized_msg = ResponseParser.normalize(message)
        if any(word in normalized_msg for word in ['reiniciar', 'recomecar', 'reset', 'restart']):
            # Se estava em crise, registra o motivo da interrupção
            if self.state == State.EMERGENCY and self.crisis_manager:
                self.log_interaction(message, "Reinicialização solicitada durante crise", False, True, 
                                   {"crisis_interrupted": True, "reason": "user_reset_request"})
            self.reset()
            return "🔄 Questionário reiniciado. Se quiser começar novamente, é só falar um Olá!\n\n"
        
        # Se está em modo de emergência/crise
        if self.state == State.EMERGENCY:
            # Garante que o crisis_manager existe
            if not self.crisis_manager:
                # Tenta carregar gerenciador existente
                self.crisis_manager = CrisisManager(self.sender_id, load_existing=True)
                # Se não conseguiu carregar do banco ou tipo está vazio, define um tipo padrão
                if not self.crisis_manager.crisis_type:
                    self.crisis_manager.crisis_type = 'unknown'
                    print(f"[Emergency] Crisis type estava None/vazio, definindo como 'unknown'")
                    # Salva o estado corrigido
                    self.crisis_manager.save_crisis_state()
                print(f"[Emergency] Gerenciador de crise carregado/criado para usuário em emergência: {self.sender_id}, tipo: {self.crisis_manager.crisis_type}")
            
            print(f"[Emergency] Processando mensagem de crise: {len(message)} caracteres")
            # Gerencia conversa de crise
            response, can_resume, crisis_metadata = self.crisis_manager.handle_crisis_conversation(message)
            
            # Log da interação
            self.log_interaction(message, response, True, True, crisis_metadata)
            
            # Se pode retomar questionário
            if can_resume:
                print(f"[Emergency] Can resume=True. Retomando questionário.")
                print(f"[Emergency] Estado anterior: {self.pre_crisis_state}, índice: {self.pre_crisis_question_index}")
                print(f"[Emergency] Resposta antes da retomada: '{response[:100]}'")
                
                self.exit_crisis_mode()
                resume_message = self.get_resume_questionnaire_message()
                
                print(f"[Emergency] Mensagem de retomada gerada: {len(resume_message)} caracteres")
                print(f"[Emergency] Primeiros 100 chars da mensagem de retomada: '{resume_message[:100]}'")
                
                # Combina resposta da crise com mensagem de retomada
                if response:
                    response += f"\n\n{resume_message}"
                else:
                    response = resume_message
                    
                print(f"[Emergency] Resposta final (primeiros 200 chars): '{response[:200]}'")
            
            return response
        
        # 1. SEMPRE faz triagem inicial com LLM
        print(f"[Safety] Iniciando triagem de segurança para: {message[:50]}...")
        screening_result = SafetyProtocol.llm_screening(message)
        
        safety_metadata = {
            'screening_model': screening_result.get('screening_model'),
            'confidence': screening_result.get('confidence'),
            'type': screening_result.get('type'),
            'detailed_check': False
        }
        
        # 2. Se detectou risco acima do limiar, faz verificação detalhada
        if screening_result.get('has_risk') and screening_result.get('confidence', 0) >= SAFETY_CONFIDENCE_THRESHOLD:
            print(f"[Safety] Risco detectado ({screening_result['type']}, confiança: {screening_result['confidence']:.2f}). Fazendo verificação detalhada...")
            
            # Verificação detalhada com modelo avançado
            detailed_result = SafetyProtocol.llm_detailed_check(message, screening_result)
            safety_metadata['detailed_check'] = True
            safety_metadata['severity'] = detailed_result.get('severity')
            safety_metadata['detailed_model'] = detailed_result.get('detailed_check_model')
            
            # Se confirmou emergência, entra em modo de crise
            if detailed_result.get('is_emergency') and detailed_result.get('confidence', 0) > 0.6:
                self.enter_crisis_mode(
                    crisis_type=detailed_result.get('type'),
                    initial_safety_score=detailed_result.get('initial_safety_score', 3)
                )
                
                # Primeira resposta da crise
                response, can_resume, crisis_metadata = self.crisis_manager.handle_crisis_conversation(message)
                safety_triggered = True
                
                # Combina metadados
                combined_metadata = {**safety_metadata, **crisis_metadata}
                
                # Log com todos os detalhes
                self.log_interaction(message, response, True, True, combined_metadata)
                return response
        
        # 3. Processa baseado no estado atual (fluxo normal)
        try:
            response = ""
            
            if self.state == State.WELCOME:
                response = self.get_welcome_message()
                self.state = State.CONSENT
                
            elif self.state == State.CONSENT:
                response = self.handle_consent(message)
                
            elif self.state == State.PHASE1_QUESTIONS:
                response, llm_used = self.handle_phase1_question(message)
                
            elif self.state == State.ASSESSMENT:
                response = self.do_assessment()
                
            elif self.state == State.FOLLOWUP_INTRO:
                # Este estado não deve mais ser usado, mas mantido por compatibilidade
                # Vai direto para FOLLOWUP_QUESTIONS
                self.state = State.FOLLOWUP_QUESTIONS
                self.current_question_index = 0
                q = FOLLOWUP_QUESTIONS[0]
                response = [
                    "Percebemos alguns sinais de risco nesta etapa. "
                    "Vamos fazer algumas perguntas de aprofundamento para entender melhor o que está acontecendo.",
                    self.format_followup_question(q, 1, 6)
                ]
                
            elif self.state == State.FOLLOWUP_QUESTIONS:
                response, llm_used = self.handle_followup_questions(message)
                
            elif self.state == State.ORIGIN_INTRO:
                # Este estado não deve mais ser usado, mas mantido por compatibilidade
                # Vai direto para ORIGIN_QUESTIONS
                self.state = State.ORIGIN_QUESTIONS
                self.current_question_index = 0
                dim = self.trigger_dimensions[0] if self.trigger_dimensions else "Risco identificado"
                dim_description = DIMENSION_DESCRIPTIONS.get(dim, dim)
                q = ORIGIN_QUESTIONS[0]
                question_text = f"🔍 *Foram encontrados riscos {dim_description}*\n\n"
                question_text += f"*1/{len(self.trigger_dimensions)*2} – {q['question']}*"
                for i, opt in enumerate(q['options'], 1):
                    question_text += f"\n{i}) {opt}"
                response = [
                    "Vamos fazer algumas outras perguntas para entender melhor a origem destes riscos.",
                    question_text
                ]
                
            elif self.state == State.ORIGIN_QUESTIONS:
                response, llm_used = self.handle_origin_questions(message)
                
            elif self.state == State.COMPLETION:
                response = self.get_completion_message()
            
            self.save_state()
            # Se response \u00e9 uma lista, converte para string para o log
            log_response = response
            if isinstance(response, list):
                log_response = " | ".join(response)  # Junta as mensagens com separador para o log
            
            # Se foi áudio, adiciona indicador no log e resposta
            log_message = message
            if audio_transcription:
                log_message = f"[ÁUDIO TRANSCRITO]: {audio_transcription}"
                # Adiciona confirmação de transcrição na resposta se não for erro/crise
                if not safety_triggered and self.state not in [State.EMERGENCY, State.COMPLETION, State.RESET]:
                    if isinstance(response, list):
                        response[0] = f"🎤 *Entendi seu áudio:* \"{audio_transcription[:100]}{'...' if len(audio_transcription) > 100 else ''}\"\n\n{response[0]}"
                    else:
                        response = f"🎤 *Entendi seu áudio:* \"{audio_transcription[:100]}{'...' if len(audio_transcription) > 100 else ''}\"\n\n{response}"
            
            self.log_interaction(log_message, log_response, llm_used, safety_triggered, safety_metadata)
            return response
            
        except Exception as e:
            print(f"[State Machine Error] {e}")
            self.log_interaction(message, "[ERROR]", False, False, safety_metadata, {"error": str(e)})
            return "Desculpe, houve um erro temporário. Pode repetir sua última mensagem?"

# =========================
# Classe para Transcrição de Áudio
# =========================
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

# =========================
# Funções auxiliares
# =========================
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

# =========================
# Lambda Handler
# =========================
def lambda_handler(event, context):
    """Handler principal da Lambda"""
    
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
