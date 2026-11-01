"""Os modelos de linguagem dos agentes de IA (fase 13, ADR 0042).

O resto do laboratório conhece só a interface ``LanguageModel`` e os tipos da conversa. O
cliente do Ollama (``ml_lab.core.llm.ollama``) não se exporta daqui de propósito: só o
``core/llm`` e os serviços o importam (há uma regra de arquitetura).
"""

from ml_lab.core.llm.base import (
    Chunk,
    LanguageModel,
    Message,
    Reply,
    Role,
    Tool,
    ToolCall,
    collect,
)
from ml_lab.core.llm.scripted import ScriptedCall, ScriptedModel

__all__ = [
    "Chunk",
    "LanguageModel",
    "Message",
    "Reply",
    "Role",
    "ScriptedCall",
    "ScriptedModel",
    "Tool",
    "ToolCall",
    "collect",
]
