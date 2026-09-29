"""Pipeline de extracción de entidades técnicas con LCEL.

    prompt | model.with_structured_output(EntidadesTecnicas) | validar_salida
    └──────────────────────────── .with_retry() ───────────────────────────┘

- `prompt`: ChatPromptTemplate modular (texto de entrada + instrucciones de formato).
- `with_structured_output`: el modelo devuelve una instancia de Pydantic, no texto crudo.
  Se usa `include_raw=True` para conservar el mensaje original y poder revisar el
  `finish_reason` antes de confiar en el objeto.
- `validar_salida`: si la respuesta vino cortada por tokens o no pasa el esquema, lanza
  una excepción y `.with_retry()` vuelve a ejecutar la cadena completa.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from schemas import EntidadesTecnicas

logger = logging.getLogger("pipeline_extraccion")

# --- Configuración del modelo -----------------

PROVIDERS = ("openai", "anthropic", "gemini")

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-flash-lite-latest",
}

DEFAULT_MAX_TOKENS = 1024
MAX_INTENTOS = 3

# Cada proveedor informa el corte por límite de tokens con un valor distinto.
FINISH_REASONS_TRUNCADO = {"length", "max_tokens", "MAX_TOKENS"}


class SalidaIncompletaError(Exception):
    """El LLM devolvió una respuesta cortada o sin la estructura esperada."""


def get_model(provider: str | None = None, max_tokens: int | None = None) -> BaseChatModel:
    """Instancia el chat model del proveedor elegido (o el de `LLM_PROVIDER`).

    Para extracción estructurada queremos determinismo, no creatividad: temperature=0.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Proveedor no soportado: {provider!r}. Opciones: {', '.join(PROVIDERS)}")

    env_var = API_KEY_ENV[provider]
    if not os.getenv(env_var):
        raise ValueError(f"Falta la variable de entorno {env_var} para usar {provider!r}")

    model = os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    # Imports diferidos: solo hace falta tener instalado el paquete del proveedor que se usa.
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0, max_tokens=max_tokens)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Los modelos Claude actuales no aceptan `temperature`; se usa el default del modelo.
        return ChatAnthropic(model=model, max_tokens=max_tokens)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=0, max_output_tokens=max_tokens)


# --- Prompt modular ---------------------------------------------------------------------

SYSTEM_PROMPT = """\
Sos un analista técnico senior. Tu tarea es extraer información estructurada del texto \
que te envía el usuario (puede ser un log de error, una descripción de arquitectura o un \
reporte de incidente).

Instrucciones de formato:
{format_instructions}"""

FORMAT_INSTRUCTIONS = """\
- tecnologias: solo las tecnologías mencionadas explícitamente en el texto, sin inventar \
ninguna, sin duplicados y con su nombre oficial. Si el texto no nombra ninguna, usá el \
componente más concreto que se mencione (p. ej. "sistema no especificado").
- nivel_de_criticidad: exactamente uno de "baja", "media" o "alta", en minúsculas.
- resumen_tecnico: 1 o 2 oraciones en español, en tono técnico, sin repetir el texto literal.
- Si el texto es ambiguo, no supongas causas que no estén escritas: decilo en el resumen."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Texto a analizar:\n<texto>\n{texto}\n</texto>"),
    ]
).partial(format_instructions=FORMAT_INSTRUCTIONS)


# --- Validación de la salida (finish_reason + esquema) ----------------------------------


def _finish_reason(raw: AIMessage) -> str | None:
    meta = raw.response_metadata or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason")
    return getattr(reason, "name", reason)  # Gemini puede devolver un Enum


def validar_salida(salida: dict) -> EntidadesTecnicas:
    """Revisa la respuesta cruda antes de aceptar el objeto parseado.

    Lanza una excepción (que dispara el reintento) si:
    1. el modelo cortó la respuesta por límite de tokens (JSON incompleto);
    2. el JSON vino mal formado o no cumple el esquema Pydantic;
    3. el modelo no devolvió ninguna estructura.
    """
    raw: AIMessage = salida["raw"]
    finish_reason = _finish_reason(raw)
    usage = raw.usage_metadata or {}
    logger.debug(
        "Respuesta cruda: finish_reason=%s, tokens_salida=%s",
        finish_reason,
        usage.get("output_tokens", "?"),
    )

    if finish_reason in FINISH_REASONS_TRUNCADO:
        raise SalidaIncompletaError(
            f"respuesta cortada por límite de tokens (finish_reason={finish_reason!r}, "
            f"tokens_salida={usage.get('output_tokens', '?')})"
        )

    if (error := salida.get("parsing_error")) is not None:
        raise error

    parsed = salida.get("parsed")
    if parsed is None:
        raise SalidaIncompletaError("el modelo no devolvió la estructura solicitada")

    # Algunos parsers devuelven dict: lo validamos igual contra el contrato.
    if not isinstance(parsed, EntidadesTecnicas):
        parsed = EntidadesTecnicas.model_validate(parsed)

    logger.debug("Salida validada contra el esquema (finish_reason=%s)", finish_reason)
    return parsed


def _resumir_error(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return str(error).splitlines()[0][:200]


# --- Logs de cada intento ---------------------------------------------------------------


class LogReintentosHandler(BaseCallbackHandler):
    """Callback que registra el inicio y el fallo de cada intento de la cadena.

    `.with_retry()` ejecuta la cadena interna una vez por intento, como hija directa del
    run de reintento, y etiqueta los reintentos con "retry:attempt:N".
    """

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id
        self.root_id: UUID | None = None
        self.intentos: dict[UUID, int] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self.root_id = run_id  # el propio RunnableRetry
            return
        if parent_run_id != self.root_id:
            return  # pasos internos (prompt, modelo, parser): no se loguean
        n = next((int(t.rsplit(":", 1)[1]) for t in tags or [] if t.startswith("retry:attempt:")), 1)
        self.intentos[run_id] = n
        if n == 1:
            logger.info("[%s] Intento %d/%d: invocando al modelo...", self.pipeline_id, n, MAX_INTENTOS)
        else:
            logger.warning(
                "[%s] Reintento %d/%d: invocando al modelo de nuevo...", self.pipeline_id, n, MAX_INTENTOS
            )

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if (n := self.intentos.get(run_id)) is not None:
            logger.warning(
                "[%s] Intento %d/%d falló (%s): %s",
                self.pipeline_id,
                n,
                MAX_INTENTOS,
                type(error).__name__,
                _resumir_error(error),
            )


# --- Cadena LCEL ------------------------------------------------------------------------


def build_chain(model: BaseChatModel) -> Runnable[dict, EntidadesTecnicas]:
    """Ensambla prompt | modelo estructurado | validación, con reintento automático."""
    # Anthropic: salida estructurada nativa. El tool calling forzado (su default) no es
    # compatible con el thinking que traen activado los modelos Claude actuales.
    kwargs = {"method": "json_schema"} if model._llm_type == "anthropic-chat" else {}
    structured_model = model.with_structured_output(EntidadesTecnicas, include_raw=True, **kwargs)

    chain = prompt | structured_model | RunnableLambda(validar_salida, name="validar_salida")

    return chain.with_retry(
        # Solo se reintentan errores de formato/contenido. Los errores de red o rate limit
        # ya los reintenta el SDK de cada proveedor, y una API key inválida no se arregla
        # reintentando.
        retry_if_exception_type=(SalidaIncompletaError, OutputParserException, ValidationError),
        stop_after_attempt=MAX_INTENTOS,
        wait_exponential_jitter=True,
    )


async def process_text(
    text: str,
    provider: str | None = None,
    *,
    model: BaseChatModel | None = None,
) -> EntidadesTecnicas:
    """Extrae las entidades técnicas de `text` y devuelve un objeto validado.

    `provider` elige el proveedor (por defecto `LLM_PROVIDER`). `model` permite inyectar
    un chat model ya configurado (por ejemplo, uno falso para las pruebas sin API key).
    Si fallan todos los reintentos, se registra el error y se relanza la excepción.
    """
    if not text or not text.strip():
        raise ValueError("El texto de entrada está vacío")

    model = model or get_model(provider)
    chain = build_chain(model)
    run_id = uuid.uuid4().hex[:6]
    nombre = getattr(model, "model_name", None) or getattr(model, "model", type(model).__name__)

    logger.info("[%s] Procesando texto (%d caracteres) con %s", run_id, len(text), nombre)
    inicio = time.perf_counter()
    try:
        resultado = await chain.ainvoke(
            {"texto": text.strip()}, config={"callbacks": [LogReintentosHandler(run_id)]}
        )
    except Exception as exc:
        logger.error(
            "[%s] El pipeline falló: %s: %s",
            run_id,
            type(exc).__name__,
            _resumir_error(exc),
        )
        raise

    logger.info(
        "[%s] Extracción validada en %.1f s: %s",
        run_id,
        time.perf_counter() - inicio,
        resultado.model_dump_json(),
    )
    return resultado
