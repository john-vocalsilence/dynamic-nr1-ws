import json
from typing import Any, Dict
import unicodedata
from openai import OpenAI
from settings import (
    MODEL_NAME, 
    OPENAI_API_KEY, 
    SCREENING_MODEL
)

client = OpenAI(api_key=OPENAI_API_KEY)

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

