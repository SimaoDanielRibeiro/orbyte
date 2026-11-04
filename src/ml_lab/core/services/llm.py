"""O serviço dos agentes de IA: se o modelo de linguagem está pronto, e o que falta fazer.

O Ollama é opcional (ADR 0042): sem ele, o laboratório funciona como sempre e os agentes de IA
ficam desligados. O ``status`` diz à interface se os pode ligar e, se não, o que fazer, em
avisos com um código estável. Não lança erro quando o Ollama não está a correr: isso é um
estado, não uma falha do pedido. Também pergunta se o modelo pensa antes de responder: o agente
de IA mostra esse raciocínio, e o Ollama recusa o pedido a um modelo que não pensa.
"""

from dataclasses import dataclass

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import LanguageModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.llm import LanguageModel
from ml_lab.core.llm.ollama import OllamaServer
from ml_lab.core.schemas.serialization import internal
from ml_lab.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LanguageModelStatus:
    """O estado do modelo de linguagem dos agentes de IA.

    Attributes:
        ready: Se os agentes de IA podem trabalhar (o Ollama responde, tem o modelo e o modelo
            pensa antes de responder).
        running: Se o Ollama responde.
        url: Onde o Ollama devia estar.
        model: O modelo configurado (``ML_LAB_OLLAMA_MODEL``).
        installed: Se o Ollama tem o modelo configurado.
        version: A versão do Ollama, quando responde.
        models: Os modelos que o Ollama tem.
        thinks: Se o modelo pensa à parte, antes de responder (o agente de IA mostra esse
            raciocínio, e o Ollama recusa o pedido a um modelo que não pensa); ``None`` quando
            não se sabe (sem o modelo, ou num Ollama que não diz as capacidades).
        notices: O que falta fazer, com um código estável (``ollama_not_running``,
            ``model_not_installed``, ``model_cannot_think``).
    """

    ready: bool
    running: bool
    url: str
    model: str
    installed: bool
    version: str | None
    models: tuple[str, ...]
    thinks: bool | None = None
    notices: tuple[Notice, ...] = ()


class LanguageModelService:
    """O modelo de linguagem dos agentes de IA (o Ollama).

    Args:
        settings: As definições (o endereço do Ollama e o modelo).
        server: O servidor a usar; por omissão, o Ollama do endereço das definições.
    """

    def __init__(self, settings: Settings, *, server: OllamaServer | None = None) -> None:
        self._model = settings.ollama_model
        self._server = server if server is not None else OllamaServer(settings.ollama_url)

    @property
    def model_name(self) -> str:
        """O modelo configurado ("qwen3:8b"), sem perguntar nada ao Ollama."""
        return self._model

    @internal
    def model(self) -> LanguageModel:
        """O modelo de linguagem configurado, pronto a conversar (serve o serviço dos agentes).

        Raises:
            LanguageModelError: Se o Ollama não estiver pronto; a mensagem diz o que fazer.
        """
        status = self.status()
        if not status.ready:
            raise LanguageModelError(" ".join(notice.message for notice in status.notices))
        return self._server.model(self._model)

    def status(self) -> LanguageModelStatus:
        """Se os agentes de IA podem trabalhar: o Ollama, o modelo e o que falta fazer."""
        url = self._server.url
        try:
            info = self._server.info()
            installed = has_model(info.models, self._model)
            capabilities = self._server.capabilities(self._model) if installed else None
        except LanguageModelError as error:
            logger.debug("o Ollama não responde", extra={"url": url, "error": str(error)})
            return LanguageModelStatus(
                ready=False,
                running=False,
                url=url,
                model=self._model,
                installed=False,
                version=None,
                models=(),
                notices=(
                    Notice.warning(
                        "ollama_not_running",
                        f"Ollama isn't running at {url}. To turn on the AI agents, open the "
                        f"Ollama app (or run 'ollama serve') and download the model with "
                        f"'ollama pull {self._model}'. The rest of the lab works without it.",
                    ),
                ),
            )
        thinks = None if capabilities is None else "thinking" in capabilities
        notices: tuple[Notice, ...] = ()
        if not installed:
            found = ", ".join(info.models) or "none"
            notices = (
                Notice.warning(
                    "model_not_installed",
                    f"Ollama is running, but it doesn't have the model {self._model}: download "
                    f"it with 'ollama pull {self._model}' (only the first time). The models it "
                    f"has: {found}.",
                ),
            )
        elif thinks is False:
            # O agente de IA pede o raciocínio ("think"), e o Ollama recusa-o a um modelo que
            # não pensa: mais vale dizê-lo aqui do que falhar a meio de uma procura.
            notices = (
                Notice.warning(
                    "model_cannot_think",
                    f"The model {self._model} doesn't think before it answers, and the AI agent "
                    "shows that thinking. Choose a model that thinks (qwen3, deepseek-r1 or "
                    "gpt-oss, for example) in ML_LAB_OLLAMA_MODEL and download it with "
                    "'ollama pull qwen3:8b'.",
                ),
            )
        return LanguageModelStatus(
            ready=installed and thinks is not False,
            running=True,
            url=url,
            model=self._model,
            installed=installed,
            version=info.version,
            models=info.models,
            thinks=thinks,
            notices=notices,
        )


def has_model(models: tuple[str, ...], wanted: str) -> bool:
    """Se ``wanted`` está entre os modelos (sem etiqueta, o Ollama lê ``:latest``)."""
    name = wanted if ":" in wanted else f"{wanted}:latest"
    return name in models or wanted in models
